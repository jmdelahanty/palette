"""Unit tests for training registry status rendering helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.utils.check_training_registry import (
    DETECT_GATE_MAX_INTERPOLATED_RATE,
    KEYPOINT_GATE_MIN_RATE,
    RecordingVocabRow,
    _detect_quality_exclusion_reason,
    _fetch_exports,
    _load_dataset_rows,
    _load_recording_rows,
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
    _summarize_detect_performance_rows,
    _summarize_detect_quality_rows,
    _summarize_eye_mask_performance_rows,
    _summarize_eye_mask_profile_rows,
    _summarize_keypoint_profile_rows,
    _summarize_keypoint_quality_rows,
)


def test_load_dataset_rows_prefers_canonical_recording_context(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        registry.upsert_dataset(
            dataset_id="dataset_ctx",
            session_uuid="session_ctx",
            zarr_path=tmp_path / "dataset_ctx.zarr",
            recording_id="recording_ctx",
            artifact_kind="source_recording",
        )
        registry.upsert_provenance(
            "dataset_ctx",
            provenance={"snapshot_status": "complete"},
            context={},
            protocol_name=None,
            protocol_hash="hash_ctx",
            acquisition={},
            zarr_purpose="analysis",
        )
        registry.conn.execute(
            """
            INSERT INTO recordings (
                recording_id, session_uuid, recording_name, recording_path,
                recording_type, recording_subtype, behavior_mode, artifact_schema_id,
                rig_id, arena_id, camera_id, created_utc, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
            """,
            (
                "recording_ctx",
                "session_ctx",
                "recording_ctx",
                str(tmp_path / "recordings" / "recording_ctx"),
                "behavior",
                "free",
                "free",
                "behavior_v1",
                "rig_recording",
                "arena_recording",
                "camera_recording",
            ),
        )
        registry.conn.execute(
            """
            UPDATE provenance
            SET rig_id = ?, arena_id = ?, camera_id = ?
            WHERE dataset_id = ?;
            """,
            ("rig_legacy", "arena_legacy", "camera_legacy", "dataset_ctx"),
        )
        registry.conn.commit()

        rows = _load_dataset_rows(registry, set_filter=None, limit=None)
    finally:
        registry.close()

    assert len(rows) == 1
    assert rows[0].dataset_id == "dataset_ctx"
    assert rows[0].rig_id == "rig_recording"
    assert rows[0].arena_id == "arena_recording"
    assert rows[0].camera_id == "camera_recording"


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


def test_detect_quality_view_outputs_summary_and_details(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    pass_path = tmp_path / "pass.zarr"
    exclude_path = tmp_path / "exclude.zarr"
    pass_path.mkdir(parents=True, exist_ok=True)
    exclude_path.mkdir(parents=True, exist_ok=True)
    pass_mtime = int(pass_path.stat().st_mtime_ns)
    exclude_mtime = int(exclude_path.stat().st_mtime_ns)

    registry.upsert_dataset(
        "dataset_pass",
        session_uuid="session_pass",
        zarr_path=pass_path,
        recording_id="recording_pass",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_exclude",
        session_uuid="session_exclude",
        zarr_path=exclude_path,
        recording_id="recording_exclude",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_detect_quality(
        dataset_id="dataset_pass",
        refined_run="refined_pass",
        refined_created_utc="2026-02-24T00:00:00+00:00",
        source_detect_run="detect_pass",
        detect_method="manual",
        review_state="approved",
        review_intended_use="training",
        review_reviewer=None,
        review_timestamp_utc="2026-02-24T00:01:00+00:00",
        review_resolved_group="manual",
        total_detections=100,
        real_detections=95,
        interpolated_detections=5,
        interpolated_detections_rate=0.05,
        zarr_mtime_ns=pass_mtime,
    )
    registry.upsert_detect_quality(
        dataset_id="dataset_exclude",
        refined_run="refined_exclude",
        refined_created_utc="2026-02-24T00:10:00+00:00",
        source_detect_run="detect_exclude",
        detect_method="manual",
        review_state="pending",
        review_intended_use="training",
        review_reviewer=None,
        review_timestamp_utc="2026-02-24T00:11:00+00:00",
        review_resolved_group="manual",
        total_detections=100,
        real_detections=70,
        interpolated_detections=30,
        interpolated_detections_rate=0.30,
        zarr_mtime_ns=exclude_mtime,
    )
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "detect-quality",
            "--show-detect-quality",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Detect Quality" in out
    assert "passing rows (approved/training, interpolated_rate<=0.25, fresh mtime): 1" in out
    assert "excluded rows: 1" in out
    assert "top exclusion reasons: wrong state/use=1" in out
    assert "dataset_pass" in out
    assert "dataset_exclude" in out


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


def test_keypoint_profile_summary_counts_and_buckets(tmp_path: Path) -> None:
    zarr_path = tmp_path / "kp_profile.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    mtime_ns = int(zarr_path.stat().st_mtime_ns)
    rows = [
        {
            "dataset_id": "a",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "keypoint_method": "traditional_pose",
            "profile_json": '{"ok":true}',
        },
        {
            "dataset_id": "b",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "keypoint_method": "traditional_pose",
            "profile_json": None,
        },
        {
            "dataset_id": "c",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": None,
            "keypoint_method": None,
            "profile_json": None,
        },
    ]
    summary = _summarize_keypoint_profile_rows(rows)
    assert summary.total_rows == 3
    assert summary.stale_rows == 1
    assert summary.exclusion_reasons == {
        "missing profile_json": 1,
        "missing profile_json+method": 1,
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


def test_detect_quality_summary_counts_and_buckets(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    mtime_ns = int(zarr_path.stat().st_mtime_ns)
    rows = [
        {
            "dataset_id": "a",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "review_state": "approved",
            "review_intended_use": "training",
            "interpolated_detections_rate": DETECT_GATE_MAX_INTERPOLATED_RATE - 0.01,
        },
        {
            "dataset_id": "b",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "review_state": None,
            "review_intended_use": "training",
            "interpolated_detections_rate": 0.1,
        },
        {
            "dataset_id": "c",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "review_state": "pending",
            "review_intended_use": "training",
            "interpolated_detections_rate": 0.1,
        },
        {
            "dataset_id": "d",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "review_state": "approved",
            "review_intended_use": "training",
            "interpolated_detections_rate": None,
        },
        {
            "dataset_id": "e",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "review_state": "approved",
            "review_intended_use": "training",
            "interpolated_detections_rate": DETECT_GATE_MAX_INTERPOLATED_RATE + 0.01,
        },
        {
            "dataset_id": "f",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": None,
            "review_state": "approved",
            "review_intended_use": "training",
            "interpolated_detections_rate": 0.1,
        },
    ]
    summary = _summarize_detect_quality_rows(rows)
    assert summary.total_rows == 6
    assert summary.passing_rows == 1
    assert summary.excluded_rows == 5
    assert summary.exclusion_reasons == {
        "high interpolation": 1,
        "missing interpolated rate": 1,
        "missing review": 1,
        "stale row: missing zarr_mtime_ns": 1,
        "wrong state/use": 1,
    }


def test_detect_quality_exclusion_reason_precedence(tmp_path: Path) -> None:
    zarr_path = tmp_path / "sample_precedence.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    mtime_ns = int(zarr_path.stat().st_mtime_ns)
    assert (
        _detect_quality_exclusion_reason(
            {
                "zarr_path": str(zarr_path),
                "zarr_mtime_ns": None,
                "review_state": "pending",
                "review_intended_use": "training",
                "interpolated_detections_rate": 0.9,
            },
            mtime_cache={},
        )
        == "stale row: missing zarr_mtime_ns"
    )
    assert (
        _detect_quality_exclusion_reason(
            {
                "zarr_path": str(zarr_path),
                "zarr_mtime_ns": mtime_ns,
                "review_state": "approved",
                "review_intended_use": "training",
                "interpolated_detections_rate": None,
            },
            mtime_cache={},
        )
        == "missing interpolated rate"
    )


def test_keypoint_profile_view_outputs_summary(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "dataset_profile_training.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    current_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    stale_mtime_ns = current_mtime_ns + 1
    registry.upsert_dataset(
        "dataset_profile",
        session_uuid="session_profile",
        zarr_path=zarr_path,
        recording_id="recording_profile",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.conn.execute(
        """
        INSERT INTO keypoint_data_profile (
            dataset_id,
            profile_run,
            recording_id,
            zarr_use,
            keypoint_method,
            source_keypoint_path,
            source_keypoint_run,
            skeleton_id,
            kpt_shape,
            profile_created_utc,
            zarr_mtime_ns,
            updated_utc,
            rows_total,
            rows_usable,
            usable_keypoints_total,
            usable_rate,
            confidence_valid_rate,
            geometry_valid_rate,
            triangle_area_p10,
            triangle_area_p50,
            triangle_area_p90,
            min_angle_p10,
            min_angle_p50,
            min_angle_p90,
            heading_p10,
            heading_p50,
            heading_p90,
            rig_id,
            camera_id,
            arena_id,
            dish_design,
            canvas_name,
            protocol_name,
            genotype,
            dpf_at_acquisition,
            profile_json
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'),
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
        """,
        (
            "dataset_profile",
            "kp_profile_001",
            "recording_profile",
            "training",
            "traditional_pose",
            "refined_keypoint_runs/refined_001/traditional_pose",
            "keypoint_001",
            "fish_v1",
            "[3,3]",
            "2026-02-24T00:00:00+00:00",
            stale_mtime_ns,
            100,
            90,
            270,
            0.9,
            0.95,
            0.96,
            0.01,
            0.02,
            0.03,
            10.0,
            20.0,
            30.0,
            -0.4,
            0.0,
            0.4,
            "omnifin0",
            "2010094",
            "arena_2",
            "cedar",
            "shadow",
            "DefaultScreen",
            "Tg(elavl3:gcamp7f)",
            7,
            None,
        ),
    )
    registry.conn.commit()
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "keypoint-profile",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Keypoint Profile" in out
    assert "total rows: 1" in out
    assert "stale rows (mtime mismatch/missing): 1" in out
    assert "top exclusion-ish reasons: missing profile_json=1" in out


def test_eye_mask_performance_summary_counts_and_rollups(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_mask_perf.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    mtime_ns = int(zarr_path.stat().st_mtime_ns)
    rows = [
        {
            "dataset_id": "a",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "review_state": "approved",
            "review_intended_use": "training",
            "source_keypoint_stale_state": None,
            "lifecycle_state": "approved",
            "successful_roi_pair_rate": 0.98,
        },
        {
            "dataset_id": "b",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "review_state": "needs_review",
            "review_intended_use": "training",
            "source_keypoint_stale_state": None,
            "lifecycle_state": "in_progress",
            "successful_roi_pair_rate": 0.90,
        },
        {
            "dataset_id": "c",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "review_state": "approved",
            "review_intended_use": "training",
            "source_keypoint_stale_state": "stale",
            "lifecycle_state": "stale",
            "successful_roi_pair_rate": 0.85,
        },
    ]
    summary = _summarize_eye_mask_performance_rows(rows)
    assert summary.total_rows == 3
    assert summary.passing_rows == 1
    assert summary.excluded_rows == 2
    assert summary.stale_rows == 1
    assert summary.exclusion_reasons == {
        "stale source keypoint": 1,
        "wrong state/use": 1,
    }
    assert summary.review_rollups == {
        "approved/training": 2,
        "needs_review/training": 1,
    }


def test_eye_mask_profile_summary_counts_and_buckets(tmp_path: Path) -> None:
    zarr_path = tmp_path / "eye_mask_profile.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    mtime_ns = int(zarr_path.stat().st_mtime_ns)
    rows = [
        {
            "dataset_id": "a",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "eye_mask_method": "refine_eye_masks",
            "review_state": "approved",
            "review_intended_use": "training",
            "profile_json": '{"ok":true}',
        },
        {
            "dataset_id": "b",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "eye_mask_method": "refine_eye_masks",
            "review_state": "pending",
            "review_intended_use": "training",
            "profile_json": '{"ok":true}',
        },
        {
            "dataset_id": "c",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns + 1,
            "eye_mask_method": "refine_eye_masks",
            "review_state": "approved",
            "review_intended_use": "training",
            "profile_json": '{"ok":true}',
        },
    ]
    summary = _summarize_eye_mask_profile_rows(rows)
    assert summary.total_rows == 3
    assert summary.stale_rows == 1
    assert summary.exclusion_reasons == {
        "stale row: mtime mismatch": 1,
        "wrong state/use": 1,
    }
    assert summary.review_rollups == {
        "approved/training": 2,
        "pending/training": 1,
    }


def test_eye_mask_performance_view_outputs_summary_and_details(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    pass_path = tmp_path / "eye_mask_pass.zarr"
    stale_path = tmp_path / "eye_mask_stale.zarr"
    pass_path.mkdir(parents=True, exist_ok=True)
    stale_path.mkdir(parents=True, exist_ok=True)
    pass_mtime = int(pass_path.stat().st_mtime_ns)
    stale_mtime = int(stale_path.stat().st_mtime_ns)
    registry.upsert_dataset(
        "dataset_pass",
        session_uuid="session_pass",
        zarr_path=pass_path,
        recording_id="recording_pass",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_stale",
        session_uuid="session_stale",
        zarr_path=stale_path,
        recording_id="recording_stale",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_eye_mask_performance(
        dataset_id="dataset_pass",
        stage_group="refined_eye_masks_runs",
        run_name="refined_eye_masks_pass",
        run_created_utc="2026-02-24T00:00:00+00:00",
        recording_id="recording_pass",
        zarr_use="training",
        method="refine_eye_masks",
        source_crop_run="crop_pass",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="kp_pass",
        source_eye_masks_run="eye_masks_pass",
        source_eye_masks_method="traditional_eye_segmentation",
        total_rois=200,
        successful_eyes=392,
        successful_roi_pairs=196,
        successful_roi_pair_rate=0.98,
        duration_seconds=40.0,
        rois_per_second=5.0,
        inference_duration_seconds=None,
        inference_average_fps=5.0,
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="alice",
        review_timestamp_utc="2026-02-24T00:10:00+00:00",
        source_keypoint_stale_state=None,
        source_keypoint_stale_reason=None,
        source_keypoint_stale_timestamp_utc=None,
        source_keypoint_stale_json=None,
        lifecycle_state="approved",
        lifecycle_reason="approved",
        zarr_mtime_ns=pass_mtime,
    )
    registry.upsert_eye_mask_performance(
        dataset_id="dataset_stale",
        stage_group="refined_eye_masks_runs",
        run_name="refined_eye_masks_stale",
        run_created_utc="2026-02-24T00:05:00+00:00",
        recording_id="recording_stale",
        zarr_use="training",
        method="refine_eye_masks",
        source_crop_run="crop_stale",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="kp_stale",
        source_eye_masks_run="eye_masks_stale",
        source_eye_masks_method="traditional_eye_segmentation",
        total_rois=200,
        successful_eyes=360,
        successful_roi_pairs=180,
        successful_roi_pair_rate=0.9,
        duration_seconds=50.0,
        rois_per_second=4.0,
        inference_duration_seconds=None,
        inference_average_fps=4.0,
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state="needs_review",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="bob",
        review_timestamp_utc="2026-02-24T00:15:00+00:00",
        source_keypoint_stale_state="stale",
        source_keypoint_stale_reason="keypoint_manual_correction",
        source_keypoint_stale_timestamp_utc="2026-02-24T00:20:00+00:00",
        source_keypoint_stale_json='{"state":"stale"}',
        lifecycle_state="stale",
        lifecycle_reason="keypoint_manual_correction",
        zarr_mtime_ns=stale_mtime,
    )
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "eye-mask-quality",
            "--show-eye-mask-quality",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Eye-Mask Quality" in out
    assert "passing rows (approved/training, non-stale source): 1" in out
    assert "excluded rows: 1" in out
    assert "stale rows: 1" in out
    assert "top exclusion reasons: stale source keypoint=1" in out
    assert "dataset_stale" in out
    assert "reason: stale source keypoint" in out


def test_subject_mask_component_view_projects_eye_compat_rows(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    eye_path = tmp_path / "eye_only.zarr"
    native_path = tmp_path / "native_subject.zarr"
    eye_path.mkdir(parents=True, exist_ok=True)
    native_path.mkdir(parents=True, exist_ok=True)
    eye_mtime = int(eye_path.stat().st_mtime_ns)
    native_mtime = int(native_path.stat().st_mtime_ns)

    registry.upsert_dataset(
        "dataset_eye_only",
        session_uuid="session_eye_only",
        zarr_path=eye_path,
        recording_id="recording_eye_only",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_native",
        session_uuid="session_native",
        zarr_path=native_path,
        recording_id="recording_native",
        artifact_kind="source_recording",
        zarr_use="training",
    )

    registry.upsert_eye_mask_performance(
        dataset_id="dataset_eye_only",
        stage_group="eye_masks_runs",
        run_name="eye_masks_eye_only",
        run_created_utc="2026-03-02T00:00:00+00:00",
        recording_id="recording_eye_only",
        zarr_use="training",
        method="traditional_eye_segmentation",
        source_crop_run="crop_eye_only",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="kp_eye_only",
        source_eye_masks_run=None,
        source_eye_masks_method=None,
        total_rois=100,
        successful_eyes=180,
        successful_roi_pairs=90,
        successful_roi_pair_rate=0.9,
        duration_seconds=20.0,
        rois_per_second=5.0,
        inference_duration_seconds=None,
        inference_average_fps=5.0,
        reason_counts_json=None,
        summary_statistics_json=None,
        zarr_mtime_ns=eye_mtime,
    )
    registry.upsert_eye_mask_performance(
        dataset_id="dataset_eye_only",
        stage_group="refined_eye_masks_runs",
        run_name="refined_eye_masks_eye_only",
        run_created_utc="2026-03-02T00:10:00+00:00",
        recording_id="recording_eye_only",
        zarr_use="training",
        method="refine_eye_masks",
        source_crop_run="crop_eye_only",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="kp_eye_only",
        source_eye_masks_run="eye_masks_eye_only",
        source_eye_masks_method="traditional_eye_segmentation",
        total_rois=100,
        successful_eyes=196,
        successful_roi_pairs=98,
        successful_roi_pair_rate=0.98,
        duration_seconds=10.0,
        rois_per_second=10.0,
        inference_duration_seconds=None,
        inference_average_fps=10.0,
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="alice",
        review_timestamp_utc="2026-03-02T00:11:00+00:00",
        lifecycle_state="approved",
        lifecycle_reason="approved",
        zarr_mtime_ns=eye_mtime,
    )

    registry.upsert_subject_mask_performance(
        dataset_id="dataset_native",
        stage_group="subject_mask_runs",
        run_name="subject_masks_native",
        run_created_utc="2026-03-02T01:00:00+00:00",
        recording_id="recording_native",
        zarr_use="training",
        subject_mask_method="subject_mask_threshold_lr_v1",
        label_schema_id="subject_v1_lr",
        source_crop_run="crop_native",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="kp_native",
        source_subject_mask_run=None,
        source_subject_mask_method=None,
        run_semantics="traditional_subject_body_inference",
        probability_semantics="normalized_background_diff",
        source_background_run=None,
        source_background_array=None,
        source_dish_mask_array=None,
        tuning_source=None,
        tuning_timestamp=None,
        total_rois=100,
        rows_with_any_mask=100,
        coverage_percent=100.0,
        duration_seconds=20.0,
        rois_per_second=5.0,
        available_component_count=4,
        available_components_json='["subject_body","eye_left","eye_right","swim_bladder"]',
        unavailable_components_json="[]",
        component_review_states_json='{"eye_left":{"state":"approved","intended_use":"training"},"eye_right":{"state":"approved","intended_use":"training"}}',
        eye_component_mode="lr",
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="alice",
        review_timestamp_utc="2026-03-02T01:01:00+00:00",
        lifecycle_state="approved",
        lifecycle_reason="approved",
        zarr_mtime_ns=native_mtime,
    )
    for component_name in ("eye_left", "eye_right"):
        registry.upsert_subject_mask_component_quality(
            dataset_id="dataset_native",
            stage_group="refined_subject_masks_runs",
            run_name="refined_subject_masks_native",
            component_name=component_name,
            component_family="eyes",
            run_created_utc="2026-03-02T01:05:00+00:00",
            recording_id="recording_native",
            zarr_use="training",
            subject_mask_method="refine_subject_masks",
            label_schema_id="subject_v1_lr",
            eye_component_mode="lr",
            source_subject_mask_run="subject_masks_native",
            available=1,
            review_state="approved",
            review_method="manual",
            review_intended_use="training",
            review_reviewer="alice",
            review_timestamp_utc="2026-03-02T01:06:00+00:00",
            total_rois=100,
            rows_with_component_mask=97,
            rows_with_component_mask_rate=0.97,
            lifecycle_state="approved",
            lifecycle_reason="approved",
            quality_updated_utc="2026-03-02T01:06:00+00:00",
            zarr_mtime_ns=native_mtime,
        )
    registry.upsert_eye_mask_performance(
        dataset_id="dataset_native",
        stage_group="refined_eye_masks_runs",
        run_name="refined_eye_masks_native",
        run_created_utc="2026-03-02T01:20:00+00:00",
        recording_id="recording_native",
        zarr_use="training",
        method="refine_eye_masks",
        source_crop_run="crop_native",
        source_keypoint_group="refined_keypoints_runs",
        source_keypoints_run="kp_native",
        source_eye_masks_run="eye_masks_native",
        source_eye_masks_method="traditional_eye_segmentation",
        total_rois=100,
        successful_eyes=198,
        successful_roi_pairs=99,
        successful_roi_pair_rate=0.99,
        duration_seconds=8.0,
        rois_per_second=12.5,
        inference_duration_seconds=None,
        inference_average_fps=12.5,
        reason_counts_json=None,
        summary_statistics_json=None,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        review_reviewer="alice",
        review_timestamp_utc="2026-03-02T01:21:00+00:00",
        lifecycle_state="approved",
        lifecycle_reason="approved",
        zarr_mtime_ns=native_mtime,
    )
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "subject-mask-components",
            "--show-subject-mask-components",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Subject-Mask Components (Unified Latest View)" in out
    assert (
        "legacy eye-stage rows are projected here only when a fresher "
        "subject-mask-native eye component is not available"
    ) in out
    assert "passing rows (approved/training, available, non-stale): 4" in out
    assert "component rollups: eye_left=2, eye_right=2" in out
    assert "stage rollups: refined_eye_masks_runs=2, refined_subject_masks_runs=2" in out
    assert "dataset_eye_only" in out
    assert "stage: refined_eye_masks_runs" in out
    assert "dataset_native" in out
    assert "stage: refined_subject_masks_runs" in out


def test_eye_mask_profile_view_outputs_summary_and_remediation(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "dataset_eye_profile.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    current_mtime_ns = int(zarr_path.stat().st_mtime_ns)
    stale_mtime_ns = current_mtime_ns + 1
    registry.upsert_dataset(
        "dataset_eye_profile",
        session_uuid="session_eye_profile",
        zarr_path=zarr_path,
        recording_id="recording_eye_profile",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.conn.execute(
        """
        CREATE TABLE IF NOT EXISTS eye_mask_data_profile (
            dataset_id TEXT NOT NULL,
            profile_run TEXT NOT NULL,
            recording_id TEXT,
            zarr_use TEXT,
            eye_mask_method TEXT,
            profile_created_utc TEXT,
            review_state TEXT,
            review_intended_use TEXT,
            source_keypoint_stale_state TEXT,
            zarr_mtime_ns INTEGER,
            profile_json TEXT,
            PRIMARY KEY (dataset_id, profile_run)
        );
        """
    )
    registry.conn.execute(
        """
        INSERT INTO eye_mask_data_profile (
            dataset_id,
            profile_run,
            recording_id,
            zarr_use,
            eye_mask_method,
            profile_created_utc,
            review_state,
            review_intended_use,
            source_keypoint_stale_state,
            zarr_mtime_ns,
            profile_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            "dataset_eye_profile",
            "eye_profile_001",
            "recording_eye_profile",
            "training",
            "refine_eye_masks",
            "2026-02-24T00:00:00+00:00",
            "pending",
            "training",
            None,
            stale_mtime_ns,
            None,
        ),
    )
    registry.conn.execute("DROP VIEW IF EXISTS eye_mask_data_profile_latest;")
    registry.conn.execute(
        """
        CREATE VIEW eye_mask_data_profile_latest AS
        SELECT
            dataset_id,
            profile_run,
            recording_id,
            zarr_use,
            eye_mask_method,
            profile_created_utc,
            review_state,
            review_intended_use,
            source_keypoint_stale_state,
            zarr_mtime_ns,
            profile_json
        FROM eye_mask_data_profile;
        """
    )
    registry.conn.commit()
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "eye-mask-profile",
            "--show-eye-mask-profile",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Eye-Mask Profile" in out
    assert "total rows: 1" in out
    assert "stale rows (mtime mismatch/missing): 1" in out
    assert "top exclusion reasons: stale row: mtime mismatch=1" in out
    assert "review rollups: pending/training=1" in out
    assert "scripts/py -m fisheye.registry.maintenance --refresh-eye-mask-profiles" in out
    assert "scripts/py -m fisheye.utils.sync_eye_mask_profile_registry" in out


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


def test_load_dataset_rows_prefers_dataset_context_current_for_recording_context(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_ctx",
        session_uuid="session_ctx",
        zarr_path=tmp_path / "ctx.zarr",
        recording_id="recording_ctx",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_provenance(
        "dataset_ctx",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={},
        zarr_purpose="training",
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path,
            recording_type, recording_subtype, behavior_mode, artifact_schema_id,
            rig_id, arena_id, camera_id, created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "recording_ctx",
            "session_ctx",
            "recording_ctx",
            str(tmp_path / "recordings" / "recording_ctx"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
            "rig_recording",
            "arena_recording",
            "camera_recording",
        ),
    )
    registry.conn.execute(
        """
        UPDATE provenance
        SET rig_id = ?, arena_id = ?, camera_id = ?
        WHERE dataset_id = ?;
        """,
        ("rig_legacy", "arena_legacy", "camera_legacy", "dataset_ctx"),
    )
    registry.conn.commit()

    rows = _load_dataset_rows(registry)
    registry.close()

    assert len(rows) == 1
    row = rows[0]
    assert row.dataset_id == "dataset_ctx"
    assert row.rig_id == "rig_recording"
    assert row.arena_id == "arena_recording"
    assert row.camera_id == "camera_recording"


def test_load_recording_rows_uses_dataset_context_current_fallback_for_dish_design(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_ctx",
        session_uuid="session_ctx",
        zarr_path=tmp_path / "ctx.zarr",
        recording_id="recording_ctx",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    registry.upsert_provenance(
        "dataset_ctx",
        provenance={},
        context={},
        protocol_name=None,
        protocol_hash=None,
        acquisition={"dish_design": "dish_design_legacy"},
        zarr_purpose="training",
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path,
            recording_type, recording_subtype, behavior_mode, artifact_schema_id,
            dish_design, created_utc, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            "recording_ctx",
            "session_ctx",
            "recording_ctx",
            str(tmp_path / "recordings" / "recording_ctx"),
            "behavior",
            "free",
            "free",
            "behavior_v1",
            None,
        ),
    )

    rows = _load_recording_rows(registry)
    registry.close()

    assert len(rows) == 1
    assert rows[0].recording_id == "recording_ctx"
    assert rows[0].dish_design == "dish_design_legacy"


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
                "arena_assignment",
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


def test_detect_performance_summary_counts_and_distributions(tmp_path: Path) -> None:
    zarr_path = tmp_path / "detect_perf.zarr"
    zarr_path.mkdir(parents=True, exist_ok=True)
    mtime_ns = int(zarr_path.stat().st_mtime_ns)
    rows = [
        {
            "dataset_id": "a",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "detection_method": "yolo",
            "model_name": "model_v1.pt",
            "coverage_percent": 95.0,
            "inference_average_fps": 60.0,
            "inference_avg_read_ms": 10.0,
        },
        {
            "dataset_id": "b",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns,
            "detection_method": "yolo",
            "model_name": "model_v2.pt",
            "coverage_percent": 85.0,
            "inference_average_fps": 40.0,
            "inference_avg_read_ms": 20.0,
        },
        {
            "dataset_id": "c",
            "zarr_path": str(zarr_path),
            "zarr_mtime_ns": mtime_ns + 999,  # mismatched -> stale
            "detection_method": "threshold",
            "model_name": "unknown",
            "coverage_percent": 70.0,
            "inference_average_fps": 80.0,
            "inference_avg_read_ms": 5.0,
        },
    ]
    summary = _summarize_detect_performance_rows(rows)
    assert summary.total_rows == 3
    assert summary.stale_rows == 1
    assert abs(summary.coverage_avg - 83.333) < 0.1
    assert summary.coverage_min == 70.0
    assert summary.coverage_max == 95.0
    assert abs(summary.fps_avg - 60.0) < 0.1
    assert summary.fps_min == 40.0
    assert summary.fps_max == 80.0
    assert abs(summary.read_ms_avg - 11.667) < 0.1
    assert summary.read_ms_min == 5.0
    assert summary.read_ms_max == 20.0
    assert summary.method_counts == {"yolo": 2, "threshold": 1}
    assert summary.model_counts == {"model_v1.pt": 1, "model_v2.pt": 1, "unknown": 1}


def test_detect_performance_view_outputs_summary_and_details(tmp_path: Path, capsys) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_a = tmp_path / "rec_a.zarr"
    zarr_b = tmp_path / "rec_b.zarr"
    zarr_a.mkdir(parents=True, exist_ok=True)
    zarr_b.mkdir(parents=True, exist_ok=True)
    mtime_a = int(zarr_a.stat().st_mtime_ns)
    mtime_b = int(zarr_b.stat().st_mtime_ns)
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=zarr_a,
        recording_id="recording_a",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_b",
        zarr_path=zarr_b,
        recording_id="recording_b",
        artifact_kind="source_recording",
        zarr_use="analysis",
    )
    registry.upsert_detect_performance(
        dataset_id="dataset_a",
        detect_run="detect_a",
        detect_created_utc="2026-02-24T00:00:00+00:00",
        recording_id="recording_a",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id=None,
        model_set_id=None,
        model_path="/models/v1.pt",
        model_name="v1.pt",
        coverage_percent=92.5,
        frames_with_detections=925,
        frames_zero_detections=75,
        total_frames=1000,
        mean_confidence=0.85,
        min_confidence=0.3,
        max_confidence=0.99,
        inference_duration_seconds=120.0,
        inference_average_fps=55.0,
        inference_avg_batch_ms=15.0,
        inference_avg_read_ms=8.0,
        conf_threshold=0.25,
        iou_threshold=0.45,
        batch_size=32,
        inference_width=640,
        inference_height=640,
        zarr_mtime_ns=mtime_a,
    )
    registry.upsert_detect_performance(
        dataset_id="dataset_b",
        detect_run="detect_b",
        detect_created_utc="2026-02-24T00:05:00+00:00",
        recording_id="recording_b",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id=None,
        model_set_id=None,
        model_path="/models/v2.pt",
        model_name="v2.pt",
        coverage_percent=45.0,
        frames_with_detections=450,
        frames_zero_detections=550,
        total_frames=1000,
        mean_confidence=0.72,
        min_confidence=0.2,
        max_confidence=0.95,
        inference_duration_seconds=200.0,
        inference_average_fps=30.0,
        inference_avg_batch_ms=25.0,
        inference_avg_read_ms=18.0,
        conf_threshold=0.25,
        iou_threshold=0.45,
        batch_size=32,
        inference_width=640,
        inference_height=640,
        zarr_mtime_ns=mtime_b,
    )
    registry.close()

    rc = check_training_registry_main(
        [
            "--registry",
            str(tmp_path / "registry.sqlite"),
            "--view",
            "detect-performance",
            "--show-detect-performance",
            "--no-rich",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Detect Performance" in out
    assert "total rows: 2" in out
    assert "stale rows: 0" in out
    assert "coverage: avg=" in out
    assert "fps: avg=" in out
    assert "read_ms: avg=" in out
    assert "methods: yolo=2" in out
    assert "dataset_a" in out
    assert "dataset_b" in out
    assert "coverage: 92.5%" in out
    assert "coverage: 45.0%" in out
    assert "model: v1.pt" in out
    assert "model: v2.pt" in out
