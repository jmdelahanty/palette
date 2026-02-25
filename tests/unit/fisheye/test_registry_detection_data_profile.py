"""Unit tests for detection-data profile registry schema and query surfaces."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry


def _upsert_profile(
    registry: Registry,
    *,
    dataset_id: str,
    profile_run: str,
    recording_id: str,
    zarr_use: str = "training",
    detection_type: str = "manual",
    detection_path: str = "refined_detect_runs/refined_001/manual",
    profile_created_utc: str = "2026-02-24T00:00:00+00:00",
    coverage_percent: float = 90.0,
    detections_total: int = 100,
    updated_utc: str | None = None,
) -> None:
    registry.upsert_detection_data_profile(
        dataset_id=dataset_id,
        profile_run=profile_run,
        recording_id=recording_id,
        zarr_use=zarr_use,
        detection_type=detection_type,
        detection_path=detection_path,
        profile_created_utc=profile_created_utc,
        frames_total=200,
        frames_with_detections=180,
        coverage_percent=coverage_percent,
        detections_total=detections_total,
        detections_per_frame_p50=1.0,
        detections_per_frame_p90=2.0,
        w_p10=0.08,
        w_p50=0.15,
        w_p90=0.22,
        h_p10=0.06,
        h_p50=0.11,
        h_p90=0.19,
        area_p10=0.006,
        area_p50=0.014,
        area_p90=0.031,
        aspect_ratio_p10=0.7,
        aspect_ratio_p50=1.3,
        aspect_ratio_p90=2.1,
        edge_proximity_rate=0.05,
        rig_id="omnifin0",
        camera_id="2010094",
        arena_id="arena_2",
        dish_design="cedar",
        canvas_name="shadow",
        protocol_name="DefaultScreen",
        genotype="Tg(elavl3:gcamp7f)",
        dpf_at_acquisition=7,
        profile_json='{"schema_name":"detection_dataset_profile","schema_version":"v1"}',
        zarr_mtime_ns=123,
        updated_utc=updated_utc,
    )


def _insert_keypoint_profile(
    registry: Registry,
    *,
    dataset_id: str,
    profile_run: str,
    recording_id: str,
    keypoint_method: str = "traditional_pose",
    profile_created_utc: str = "2026-02-24T00:00:00+00:00",
    usable_rate: float = 0.90,
) -> None:
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
        )
        ON CONFLICT(dataset_id, profile_run) DO UPDATE SET
            recording_id=excluded.recording_id,
            zarr_use=excluded.zarr_use,
            keypoint_method=excluded.keypoint_method,
            source_keypoint_path=excluded.source_keypoint_path,
            source_keypoint_run=excluded.source_keypoint_run,
            skeleton_id=excluded.skeleton_id,
            kpt_shape=excluded.kpt_shape,
            profile_created_utc=excluded.profile_created_utc,
            zarr_mtime_ns=excluded.zarr_mtime_ns,
            updated_utc=datetime('now'),
            rows_total=excluded.rows_total,
            rows_usable=excluded.rows_usable,
            usable_keypoints_total=excluded.usable_keypoints_total,
            usable_rate=excluded.usable_rate,
            confidence_valid_rate=excluded.confidence_valid_rate,
            geometry_valid_rate=excluded.geometry_valid_rate,
            triangle_area_p10=excluded.triangle_area_p10,
            triangle_area_p50=excluded.triangle_area_p50,
            triangle_area_p90=excluded.triangle_area_p90,
            min_angle_p10=excluded.min_angle_p10,
            min_angle_p50=excluded.min_angle_p50,
            min_angle_p90=excluded.min_angle_p90,
            heading_p10=excluded.heading_p10,
            heading_p50=excluded.heading_p50,
            heading_p90=excluded.heading_p90,
            rig_id=excluded.rig_id,
            camera_id=excluded.camera_id,
            arena_id=excluded.arena_id,
            dish_design=excluded.dish_design,
            canvas_name=excluded.canvas_name,
            protocol_name=excluded.protocol_name,
            genotype=excluded.genotype,
            dpf_at_acquisition=excluded.dpf_at_acquisition,
            profile_json=excluded.profile_json
        ;
        """,
        (
            dataset_id,
            profile_run,
            recording_id,
            "training",
            keypoint_method,
            f"refined_keypoint_runs/{profile_run}/{keypoint_method}",
            f"keypoint_{profile_run}",
            "fish_v1",
            "[3,3]",
            profile_created_utc,
            123,
            200,
            180,
            540,
            usable_rate,
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
            '{"schema_name":"keypoint_dataset_profile","schema_version":"v1"}',
        ),
    )
    registry.conn.commit()


def _upsert_eye_mask_profile(
    registry: Registry,
    *,
    dataset_id: str,
    profile_run: str,
    recording_id: str,
    stage_group: str = "refined_eye_masks_runs",
    eye_mask_method: str = "traditional",
    profile_created_utc: str = "2026-02-24T00:00:00+00:00",
    usable_rate: float = 0.90,
) -> None:
    registry.upsert_eye_mask_data_profile(
        dataset_id=dataset_id,
        profile_run=profile_run,
        recording_id=recording_id,
        zarr_use="training",
        stage_group=stage_group,
        eye_mask_method=eye_mask_method,
        source_eye_mask_path=f"{stage_group}/{profile_run}",
        source_eye_mask_run=profile_run,
        source_keypoint_path="refined_keypoints_runs/refined_keypoints_001",
        source_keypoint_run="refined_keypoints_001",
        source_crop_run="crop_001",
        profile_created_utc=profile_created_utc,
        rows_total=200,
        rows_usable=180,
        usable_rate=usable_rate,
        reviewed_rate=1.0,
        excluded_rate=0.10,
        exclusion_reasons_json='{"ellipse_fit_failed":20}',
        ellipse_success_rate=0.95,
        pair_success_rate=0.90,
        area_p10=300.0,
        area_p50=420.0,
        area_p90=560.0,
        left_area_p10=145.0,
        left_area_p50=205.0,
        left_area_p90=265.0,
        right_area_p10=150.0,
        right_area_p50=210.0,
        right_area_p90=275.0,
        union_area_p10=300.0,
        union_area_p50=415.0,
        union_area_p90=545.0,
        area_lr_ratio_p10=0.90,
        area_lr_ratio_p50=0.98,
        area_lr_ratio_p90=1.08,
        major_axis_p10=14.0,
        major_axis_p50=19.0,
        major_axis_p90=25.0,
        minor_axis_p10=7.0,
        minor_axis_p50=10.0,
        minor_axis_p90=13.0,
        aspect_ratio_p10=1.4,
        aspect_ratio_p50=1.8,
        aspect_ratio_p90=2.2,
        eye_separation_p10=21.0,
        eye_separation_p50=27.0,
        eye_separation_p90=34.0,
        edge_proximity_rate=0.05,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        review_timestamp_utc="2026-02-24T00:05:00+00:00",
        source_keypoint_stale_state="fresh",
        source_keypoint_stale_reason=None,
        source_keypoint_stale_timestamp_utc="2026-02-24T00:04:00+00:00",
        source_keypoint_stale_json='{"state":"fresh"}',
        rig_id="omnifin0",
        camera_id="2010094",
        arena_id="arena_2",
        dish_design="cedar",
        canvas_name="shadow",
        protocol_name="DefaultScreen",
        genotype="Tg(elavl3:gcamp7f)",
        dpf_at_acquisition=7,
        profile_json='{"schema_name":"eye_mask_dataset_profile","schema_version":"v1"}',
        zarr_mtime_ns=123,
    )


def test_schema_has_detection_data_profile_table_views_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")

    table = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'detection_data_profile';
        """
    ).fetchone()
    assert table is not None

    views = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name IN (
            'detection_data_profile_latest',
            'recording_detection_data_profile_latest'
        );
        """
    ).fetchall()
    assert {str(row["name"]) for row in views} == {
        "detection_data_profile_latest",
        "recording_detection_data_profile_latest",
    }

    indexes = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (
            'idx_detection_data_profile_recording_created',
            'idx_detection_data_profile_detection_scope',
            'idx_detection_data_profile_coverage'
        );
        """
    ).fetchall()
    assert {str(row["name"]) for row in indexes} == {
        "idx_detection_data_profile_recording_created",
        "idx_detection_data_profile_detection_scope",
        "idx_detection_data_profile_coverage",
    }
    registry.close()


def test_schema_has_keypoint_data_profile_table_views_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")

    table = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'keypoint_data_profile';
        """
    ).fetchone()
    assert table is not None

    views = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name IN (
            'keypoint_data_profile_latest',
            'recording_keypoint_data_profile_latest'
        );
        """
    ).fetchall()
    assert {str(row["name"]) for row in views} == {
        "keypoint_data_profile_latest",
        "recording_keypoint_data_profile_latest",
    }

    indexes = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (
            'idx_keypoint_data_profile_recording_created',
            'idx_keypoint_data_profile_method_scope',
            'idx_keypoint_data_profile_method_usable_rate'
        );
        """
    ).fetchall()
    assert {str(row["name"]) for row in indexes} == {
        "idx_keypoint_data_profile_recording_created",
        "idx_keypoint_data_profile_method_scope",
        "idx_keypoint_data_profile_method_usable_rate",
    }
    registry.close()


def test_schema_has_eye_mask_data_profile_table_views_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")

    table = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'eye_mask_data_profile';
        """
    ).fetchone()
    assert table is not None

    views = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view' AND name IN (
            'eye_mask_data_profile_latest',
            'recording_eye_mask_data_profile_latest'
        );
        """
    ).fetchall()
    assert {str(row["name"]) for row in views} == {
        "eye_mask_data_profile_latest",
        "recording_eye_mask_data_profile_latest",
    }

    indexes = registry.conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'index' AND name IN (
            'idx_eye_mask_data_profile_recording_created',
            'idx_eye_mask_data_profile_method_scope',
            'idx_eye_mask_data_profile_stage_usable_rate'
        );
        """
    ).fetchall()
    assert {str(row["name"]) for row in indexes} == {
        "idx_eye_mask_data_profile_recording_created",
        "idx_eye_mask_data_profile_method_scope",
        "idx_eye_mask_data_profile_stage_usable_rate",
    }
    registry.close()


def test_query_detection_data_profile_latest_and_recording_latest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=tmp_path / "a_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_b",
        zarr_path=tmp_path / "b_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )

    _upsert_profile(
        registry,
        dataset_id="dataset_a",
        profile_run="profile_old",
        recording_id="rec_shared",
        profile_created_utc="2026-02-22T00:00:00+00:00",
        coverage_percent=75.0,
    )
    _upsert_profile(
        registry,
        dataset_id="dataset_a",
        profile_run="profile_new",
        recording_id="rec_shared",
        profile_created_utc="2026-02-23T00:00:00+00:00",
        coverage_percent=88.0,
    )
    _upsert_profile(
        registry,
        dataset_id="dataset_b",
        profile_run="profile_b",
        recording_id="rec_shared",
        profile_created_utc="2026-02-24T00:00:00+00:00",
        coverage_percent=93.0,
    )

    dataset_latest = registry.query_detection_data_profile_latest(dataset_ids=["dataset_a"])
    assert len(dataset_latest) == 1
    assert str(dataset_latest[0]["dataset_id"]) == "dataset_a"
    assert str(dataset_latest[0]["profile_run"]) == "profile_new"
    assert float(dataset_latest[0]["coverage_percent"]) == 88.0
    assert str(dataset_latest[0]["genotype"]) == "Tg(elavl3:gcamp7f)"
    assert int(dataset_latest[0]["dpf_at_acquisition"]) == 7

    coverage_filtered = registry.query_detection_data_profile_latest(
        min_coverage_percent=90.0,
        detection_type="manual",
        zarr_use="training",
    )
    assert len(coverage_filtered) == 1
    assert str(coverage_filtered[0]["dataset_id"]) == "dataset_b"

    recording_latest = registry.query_recording_detection_data_profile_latest(
        recording_ids=["rec_shared"],
        detection_type="manual",
        min_coverage_percent=80.0,
    )
    assert len(recording_latest) == 1
    assert str(recording_latest[0]["recording_id"]) == "rec_shared"
    assert str(recording_latest[0]["dataset_id"]) == "dataset_b"
    assert str(recording_latest[0]["profile_run"]) == "profile_b"
    registry.close()


def test_query_keypoint_data_profile_latest_and_recording_latest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=tmp_path / "a_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_b",
        zarr_path=tmp_path / "b_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )

    _insert_keypoint_profile(
        registry,
        dataset_id="dataset_a",
        profile_run="kp_old",
        recording_id="rec_shared",
        keypoint_method="traditional_pose",
        profile_created_utc="2026-02-22T00:00:00+00:00",
        usable_rate=0.82,
    )
    _insert_keypoint_profile(
        registry,
        dataset_id="dataset_a",
        profile_run="kp_new",
        recording_id="rec_shared",
        keypoint_method="traditional_pose",
        profile_created_utc="2026-02-23T00:00:00+00:00",
        usable_rate=0.91,
    )
    _insert_keypoint_profile(
        registry,
        dataset_id="dataset_b",
        profile_run="kp_b",
        recording_id="rec_shared",
        keypoint_method="yolo_pose",
        profile_created_utc="2026-02-24T00:00:00+00:00",
        usable_rate=0.94,
    )

    dataset_latest = registry.query_keypoint_data_profile_latest(dataset_ids=["dataset_a"])
    assert len(dataset_latest) == 1
    assert str(dataset_latest[0]["dataset_id"]) == "dataset_a"
    assert str(dataset_latest[0]["profile_run"]) == "kp_new"
    assert str(dataset_latest[0]["keypoint_method"]) == "traditional_pose"
    assert float(dataset_latest[0]["usable_rate"]) == 0.91
    assert str(dataset_latest[0]["genotype"]) == "Tg(elavl3:gcamp7f)"
    assert int(dataset_latest[0]["dpf_at_acquisition"]) == 7

    usable_filtered = registry.query_keypoint_data_profile_latest(
        min_usable_rate=0.93,
        zarr_use="training",
    )
    assert len(usable_filtered) == 1
    assert str(usable_filtered[0]["dataset_id"]) == "dataset_b"
    assert str(usable_filtered[0]["keypoint_method"]) == "yolo_pose"

    recording_latest = registry.query_recording_keypoint_data_profile_latest(
        recording_ids=["rec_shared"],
        keypoint_method="yolo_pose",
        min_usable_rate=0.90,
    )
    assert len(recording_latest) == 1
    assert str(recording_latest[0]["recording_id"]) == "rec_shared"
    assert str(recording_latest[0]["dataset_id"]) == "dataset_b"
    assert str(recording_latest[0]["profile_run"]) == "kp_b"
    registry.close()


def test_query_eye_mask_data_profile_latest_and_recording_latest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_a",
        zarr_path=tmp_path / "a_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_b",
        zarr_path=tmp_path / "b_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )

    _upsert_eye_mask_profile(
        registry,
        dataset_id="dataset_a",
        profile_run="eye_old",
        recording_id="rec_shared",
        stage_group="refined_eye_masks_runs",
        eye_mask_method="traditional",
        profile_created_utc="2026-02-22T00:00:00+00:00",
        usable_rate=0.82,
    )
    _upsert_eye_mask_profile(
        registry,
        dataset_id="dataset_a",
        profile_run="eye_new",
        recording_id="rec_shared",
        stage_group="refined_eye_masks_runs",
        eye_mask_method="traditional",
        profile_created_utc="2026-02-23T00:00:00+00:00",
        usable_rate=0.91,
    )
    _upsert_eye_mask_profile(
        registry,
        dataset_id="dataset_b",
        profile_run="eye_b",
        recording_id="rec_shared",
        stage_group="refined_eye_masks_runs",
        eye_mask_method="yolo",
        profile_created_utc="2026-02-24T00:00:00+00:00",
        usable_rate=0.94,
    )

    dataset_latest = registry.query_eye_mask_data_profile_latest(dataset_ids=["dataset_a"])
    assert len(dataset_latest) == 1
    assert str(dataset_latest[0]["dataset_id"]) == "dataset_a"
    assert str(dataset_latest[0]["profile_run"]) == "eye_new"
    assert str(dataset_latest[0]["eye_mask_method"]) == "traditional"
    assert float(dataset_latest[0]["usable_rate"]) == 0.91
    assert float(dataset_latest[0]["left_area_p50"]) == 205.0
    assert float(dataset_latest[0]["area_lr_ratio_p50"]) == 0.98
    assert str(dataset_latest[0]["genotype"]) == "Tg(elavl3:gcamp7f)"
    assert int(dataset_latest[0]["dpf_at_acquisition"]) == 7

    usable_filtered = registry.query_eye_mask_data_profile_latest(
        min_usable_rate=0.93,
        zarr_use="training",
        stage_group="refined_eye_masks_runs",
    )
    assert len(usable_filtered) == 1
    assert str(usable_filtered[0]["dataset_id"]) == "dataset_b"
    assert str(usable_filtered[0]["eye_mask_method"]) == "yolo"

    recording_latest = registry.query_recording_eye_mask_data_profile_latest(
        recording_ids=["rec_shared"],
        stage_group="refined_eye_masks_runs",
        eye_mask_method="yolo",
        min_usable_rate=0.90,
    )
    assert len(recording_latest) == 1
    assert str(recording_latest[0]["recording_id"]) == "rec_shared"
    assert str(recording_latest[0]["dataset_id"]) == "dataset_b"
    assert str(recording_latest[0]["profile_run"]) == "eye_b"
    registry.close()


def test_replace_detection_data_profile_replaces_dataset_scope_rows(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_replace",
        session_uuid="session_replace",
        zarr_path=tmp_path / "replace_training.zarr",
        recording_id="rec_replace",
        zarr_use="training",
    )

    registry.replace_detection_data_profile(
        "dataset_replace",
        [
            {
                "profile_run": "profile_a",
                "recording_id": "rec_replace",
                "zarr_use": "training",
                "detection_type": "manual",
                "detection_path": "refined_detect_runs/refined_001/manual",
                "profile_created_utc": "2026-02-21T00:00:00+00:00",
                "frames_total": 100,
                "frames_with_detections": 90,
                "coverage_percent": 90.0,
                "detections_total": 90,
                "detections_per_frame_p50": 1.0,
                "detections_per_frame_p90": 1.0,
                "w_p10": 0.08,
                "w_p50": 0.15,
                "w_p90": 0.22,
                "h_p10": 0.06,
                "h_p50": 0.11,
                "h_p90": 0.19,
                "area_p10": 0.006,
                "area_p50": 0.014,
                "area_p90": 0.031,
                "aspect_ratio_p10": 0.7,
                "aspect_ratio_p50": 1.3,
                "aspect_ratio_p90": 2.1,
                "edge_proximity_rate": 0.05,
                "rig_id": "omnifin0",
                "camera_id": "2010094",
                "arena_id": "arena_2",
                "dish_design": "cedar",
                "canvas_name": "shadow",
                "protocol_name": "DefaultScreen",
                "genotype": "Tg(elavl3:gcamp7f)",
                "dpf_at_acquisition": 7,
                "profile_json": '{"schema_name":"detection_dataset_profile","schema_version":"v1"}',
                "zarr_mtime_ns": 10,
            },
            {
                "profile_run": "profile_b",
                "recording_id": "rec_replace",
                "zarr_use": "training",
                "detection_type": "interpolated",
                "detection_path": "refined_detect_runs/refined_001/interpolated",
                "profile_created_utc": "2026-02-22T00:00:00+00:00",
                "frames_total": 100,
                "frames_with_detections": 85,
                "coverage_percent": 85.0,
                "detections_total": 95,
                "detections_per_frame_p50": 1.0,
                "detections_per_frame_p90": 2.0,
                "w_p10": 0.08,
                "w_p50": 0.15,
                "w_p90": 0.22,
                "h_p10": 0.06,
                "h_p50": 0.11,
                "h_p90": 0.19,
                "area_p10": 0.006,
                "area_p50": 0.014,
                "area_p90": 0.031,
                "aspect_ratio_p10": 0.7,
                "aspect_ratio_p50": 1.3,
                "aspect_ratio_p90": 2.1,
                "edge_proximity_rate": 0.07,
                "rig_id": "omnifin0",
                "camera_id": "2010094",
                "arena_id": "arena_2",
                "dish_design": "cedar",
                "canvas_name": "shadow",
                "protocol_name": "DefaultScreen",
                "genotype": "Tg(elavl3:gcamp7f)",
                "dpf_at_acquisition": 7,
                "profile_json": '{"schema_name":"detection_dataset_profile","schema_version":"v1"}',
                "zarr_mtime_ns": 11,
            },
        ],
    )
    count_before = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM detection_data_profile WHERE dataset_id = 'dataset_replace';"
    ).fetchone()
    assert count_before is not None
    assert int(count_before["n"]) == 2

    registry.replace_detection_data_profile(
        "dataset_replace",
        [
            {
                "profile_run": "profile_c",
                "recording_id": "rec_replace",
                "zarr_use": "training",
                "detection_type": "manual",
                "detection_path": "refined_detect_runs/refined_002/manual",
                "profile_created_utc": "2026-02-23T00:00:00+00:00",
                "frames_total": 120,
                "frames_with_detections": 110,
                "coverage_percent": 91.7,
                "detections_total": 120,
                "detections_per_frame_p50": 1.0,
                "detections_per_frame_p90": 2.0,
                "w_p10": 0.08,
                "w_p50": 0.15,
                "w_p90": 0.22,
                "h_p10": 0.06,
                "h_p50": 0.11,
                "h_p90": 0.19,
                "area_p10": 0.006,
                "area_p50": 0.014,
                "area_p90": 0.031,
                "aspect_ratio_p10": 0.7,
                "aspect_ratio_p50": 1.3,
                "aspect_ratio_p90": 2.1,
                "edge_proximity_rate": 0.04,
                "rig_id": "omnifin0",
                "camera_id": "2010094",
                "arena_id": "arena_2",
                "dish_design": "cedar",
                "canvas_name": "shadow",
                "protocol_name": "DefaultScreen",
                "genotype": "Tg(elavl3:gcamp7f)",
                "dpf_at_acquisition": 8,
                "profile_json": '{"schema_name":"detection_dataset_profile","schema_version":"v1","run":"c"}',
                "zarr_mtime_ns": 12,
            }
        ],
    )
    count_after = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM detection_data_profile WHERE dataset_id = 'dataset_replace';"
    ).fetchone()
    assert count_after is not None
    assert int(count_after["n"]) == 1

    latest = registry.query_detection_data_profile_latest(dataset_ids=["dataset_replace"])
    assert len(latest) == 1
    assert str(latest[0]["profile_run"]) == "profile_c"
    assert str(latest[0]["genotype"]) == "Tg(elavl3:gcamp7f)"
    assert int(latest[0]["dpf_at_acquisition"]) == 8
    assert '"run":"c"' in str(latest[0]["profile_json"])
    registry.close()
