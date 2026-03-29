"""Unit tests for detection-data profile registry schema and query surfaces."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry


_PROFILE_DUPLICATE_RECORDING_CONTEXT_FIELDS = (
    "rig_id",
    "camera_id",
    "arena_id",
    "dish_design",
    "canvas_name",
    "protocol_name",
)
_PROFILE_DUPLICATE_BIOLOGY_FIELDS = ("genotype", "dpf_at_acquisition")


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
    rig_id: str = "omnifin0",
    camera_id: str = "2010094",
    arena_id: str = "arena_2",
    dish_design: str = "cedar",
    canvas_name: str = "shadow",
    protocol_name: str = "DefaultScreen",
    genotype: str = "Tg(elavl3:gcamp7f)",
    dpf_at_acquisition: int = 7,
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
        rig_id=rig_id,
        camera_id=camera_id,
        arena_id=arena_id,
        dish_design=dish_design,
        canvas_name=canvas_name,
        protocol_name=protocol_name,
        genotype=genotype,
        dpf_at_acquisition=dpf_at_acquisition,
        profile_json='{"schema_name":"detection_dataset_profile","schema_version":"v1"}',
        zarr_mtime_ns=123,
        updated_utc=updated_utc,
    )


def _seed_canonical_context_for_dataset(
    registry: Registry,
    *,
    root: Path,
    dataset_id: str,
    recording_id: str,
    session_uuid: str,
    zarr_use: str = "training",
    dpf_at_acquisition: int = 7,
) -> None:
    registry.upsert_provenance(
        dataset_id,
        provenance={
            "fish_id": f"legacy_{dataset_id}",
            "dish_id": f"legacy_dish_{dataset_id}",
            "cross_id": f"legacy_cross_{dataset_id}",
            "genotype": f"legacy_genotype_{dataset_id}",
            "dpf_at_acquisition": dpf_at_acquisition + 2,
            "snapshot_status": "complete",
        },
        context={
            "rig_id": "rig_legacy",
            "arena_id": "arena_legacy",
            "camera_id": "camera_legacy",
            "canvas_name": "canvas_legacy",
        },
        protocol_name="protocol_legacy",
        protocol_hash=None,
        acquisition={"dish_design": "dish_design_legacy"},
        zarr_purpose=zarr_use,
    )
    registry.conn.execute(
        """
        INSERT INTO recordings (
            recording_id, session_uuid, recording_name, recording_path, started_utc,
            recording_type, recording_subtype, behavior_mode, artifact_schema_id,
            rig_id, arena_id, camera_id, canvas_name, protocol_name, dish_design,
            created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(recording_id) DO UPDATE SET
            session_uuid=excluded.session_uuid,
            recording_name=excluded.recording_name,
            recording_path=excluded.recording_path,
            started_utc=excluded.started_utc,
            recording_type=excluded.recording_type,
            recording_subtype=excluded.recording_subtype,
            behavior_mode=excluded.behavior_mode,
            artifact_schema_id=excluded.artifact_schema_id,
            rig_id=excluded.rig_id,
            arena_id=excluded.arena_id,
            camera_id=excluded.camera_id,
            canvas_name=excluded.canvas_name,
            protocol_name=excluded.protocol_name,
            dish_design=excluded.dish_design,
            updated_utc=datetime('now');
        """,
        (
            recording_id,
            session_uuid,
            recording_id,
            str(root / "recordings" / recording_id),
            "2026-03-13T00:00:00+00:00",
            "behavior",
            "free",
            "free",
            "behavior_v1",
            "rig_recording",
            "arena_recording",
            "camera_recording",
            "canvas_recording",
            "protocol_recording",
            "dish_design_recording",
        ),
    )
    cross_id = f"cross_{dataset_id}"
    dish_id = f"dish_{dataset_id}"
    subject_id = f"subject_{dataset_id}"
    genotype = f"genotype_{dataset_id}"
    registry.conn.execute(
        """
        INSERT INTO crosses (cross_id, genotype, line_strain, created_utc, updated_utc)
        VALUES (?, ?, ?, datetime('now'), datetime('now'));
        """,
        (cross_id, genotype, f"line_{dataset_id}"),
    )
    registry.conn.execute(
        """
        INSERT INTO dishes (dish_id, cross_id, species, created_utc, updated_utc)
        VALUES (?, ?, ?, datetime('now'), datetime('now'));
        """,
        (dish_id, cross_id, "danio_rerio"),
    )
    registry.conn.execute(
        """
        INSERT INTO subjects (subject_id, dish_id, species, sex, created_utc, updated_utc)
        VALUES (?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (subject_id, dish_id, "danio_rerio", "unknown"),
    )
    registry.conn.execute(
        """
        INSERT INTO recording_subjects (
            recording_id, subject_id, dataset_id, dish_id, cross_id, dpf_at_acquisition,
            genotype, line_strain, species, sex, created_utc, updated_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'));
        """,
        (
            recording_id,
            subject_id,
            dataset_id,
            dish_id,
            cross_id,
            dpf_at_acquisition,
            genotype,
            f"line_{dataset_id}",
            "danio_rerio",
            "unknown",
        ),
    )
    registry.conn.commit()


def _fetch_profile_row(
    registry: Registry,
    *,
    table_name: str,
    dataset_id: str,
    profile_run: str,
):
    row = registry.conn.execute(
        f"""
        SELECT *
        FROM {table_name}
        WHERE dataset_id = ? AND profile_run = ?;
        """,
        (dataset_id, profile_run),
    ).fetchone()
    assert row is not None
    return row


def _assert_profile_duplicate_context_values(
    row,
    *,
    rig_id,
    camera_id,
    arena_id,
    dish_design,
    canvas_name,
    protocol_name,
    genotype,
    dpf_at_acquisition,
) -> None:
    assert row["rig_id"] == rig_id
    assert row["camera_id"] == camera_id
    assert row["arena_id"] == arena_id
    assert row["dish_design"] == dish_design
    assert row["canvas_name"] == canvas_name
    assert row["protocol_name"] == protocol_name
    assert row["genotype"] == genotype
    assert row["dpf_at_acquisition"] == dpf_at_acquisition


def _assert_profile_duplicate_context_cleared(row) -> None:
    for field in _PROFILE_DUPLICATE_RECORDING_CONTEXT_FIELDS + _PROFILE_DUPLICATE_BIOLOGY_FIELDS:
        assert row[field] is None


def _insert_keypoint_profile(
    registry: Registry,
    *,
    dataset_id: str,
    profile_run: str,
    recording_id: str,
    zarr_use: str = "training",
    keypoint_method: str = "traditional_pose",
    profile_created_utc: str = "2026-02-24T00:00:00+00:00",
    usable_rate: float = 0.90,
    rig_id: str = "omnifin0",
    camera_id: str = "2010094",
    arena_id: str = "arena_2",
    dish_design: str = "cedar",
    canvas_name: str = "shadow",
    protocol_name: str = "DefaultScreen",
    genotype: str = "Tg(elavl3:gcamp7f)",
    dpf_at_acquisition: int = 7,
) -> None:
    registry.upsert_keypoint_data_profile(
        dataset_id=dataset_id,
        profile_run=profile_run,
        recording_id=recording_id,
        zarr_use=zarr_use,
        keypoint_method=keypoint_method,
        source_keypoint_path=f"refined_keypoint_runs/{profile_run}/{keypoint_method}",
        source_keypoint_run=f"keypoint_{profile_run}",
        skeleton_id="fish_v1",
        kpt_shape="[3,3]",
        profile_created_utc=profile_created_utc,
        rows_total=200,
        rows_usable=180,
        usable_keypoints_total=540,
        usable_rate=usable_rate,
        confidence_valid_rate=0.95,
        geometry_valid_rate=0.96,
        triangle_area_p10=0.01,
        triangle_area_p50=0.02,
        triangle_area_p90=0.03,
        min_angle_p10=10.0,
        min_angle_p50=20.0,
        min_angle_p90=30.0,
        heading_p10=-0.4,
        heading_p50=0.0,
        heading_p90=0.4,
        rig_id=rig_id,
        camera_id=camera_id,
        arena_id=arena_id,
        dish_design=dish_design,
        canvas_name=canvas_name,
        protocol_name=protocol_name,
        genotype=genotype,
        dpf_at_acquisition=dpf_at_acquisition,
        profile_json='{"schema_name":"keypoint_dataset_profile","schema_version":"v1"}',
        zarr_mtime_ns=123,
    )


def _upsert_eye_mask_profile(
    registry: Registry,
    *,
    dataset_id: str,
    profile_run: str,
    recording_id: str,
    zarr_use: str = "training",
    stage_group: str = "refined_eye_masks_runs",
    eye_mask_method: str = "traditional",
    profile_created_utc: str = "2026-02-24T00:00:00+00:00",
    usable_rate: float = 0.90,
    rig_id: str = "omnifin0",
    camera_id: str = "2010094",
    arena_id: str = "arena_2",
    dish_design: str = "cedar",
    canvas_name: str = "shadow",
    protocol_name: str = "DefaultScreen",
    genotype: str = "Tg(elavl3:gcamp7f)",
    dpf_at_acquisition: int = 7,
) -> None:
    registry.upsert_eye_mask_data_profile(
        dataset_id=dataset_id,
        profile_run=profile_run,
        recording_id=recording_id,
        zarr_use=zarr_use,
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
        rig_id=rig_id,
        camera_id=camera_id,
        arena_id=arena_id,
        dish_design=dish_design,
        canvas_name=canvas_name,
        protocol_name=protocol_name,
        genotype=genotype,
        dpf_at_acquisition=dpf_at_acquisition,
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
        session_uuid="session_shared",
        zarr_path=tmp_path / "a_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_shared",
        zarr_path=tmp_path / "b_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_a",
        recording_id="rec_shared",
        session_uuid="session_shared",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_b",
        recording_id="rec_shared",
        session_uuid="session_shared",
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
    assert dataset_latest[0]["genotype"] is None
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
    assert recording_latest[0]["genotype"] is None
    assert int(recording_latest[0]["dpf_at_acquisition"]) == 7
    registry.close()


def test_detection_profile_views_prefer_dataset_context_current(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_ctx",
        session_uuid="session_ctx",
        zarr_path=tmp_path / "ctx_training.zarr",
        recording_id="recording_ctx",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_ctx",
        recording_id="recording_ctx",
        session_uuid="session_ctx",
    )
    _upsert_profile(
        registry,
        dataset_id="dataset_ctx",
        profile_run="profile_ctx",
        recording_id="recording_stale",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )

    dataset_latest = registry.query_detection_data_profile_latest(dataset_ids=["dataset_ctx"])
    assert len(dataset_latest) == 1
    row = dataset_latest[0]
    assert str(row["recording_id"]) == "recording_ctx"
    assert str(row["zarr_use"]) == "training"
    assert str(row["rig_id"]) == "rig_recording"
    assert str(row["camera_id"]) == "camera_recording"
    assert str(row["arena_id"]) == "arena_recording"
    assert str(row["dish_design"]) == "dish_design_recording"
    assert str(row["canvas_name"]) == "canvas_recording"
    assert str(row["protocol_name"]) == "protocol_recording"
    assert str(row["genotype"]) == "genotype_dataset_ctx"
    assert int(row["dpf_at_acquisition"]) == 7

    training_filtered = registry.query_detection_data_profile_latest(
        dataset_ids=["dataset_ctx"],
        zarr_use="training",
    )
    assert len(training_filtered) == 1
    analysis_filtered = registry.query_detection_data_profile_latest(
        dataset_ids=["dataset_ctx"],
        zarr_use="analysis",
    )
    assert analysis_filtered == []

    recording_latest = registry.query_recording_detection_data_profile_latest(
        recording_ids=["recording_ctx"]
    )
    assert len(recording_latest) == 1
    recording_row = recording_latest[0]
    assert str(recording_row["dataset_id"]) == "dataset_ctx"
    assert str(recording_row["recording_id"]) == "recording_ctx"
    assert str(recording_row["rig_id"]) == "rig_recording"
    assert str(recording_row["genotype"]) == "genotype_dataset_ctx"
    assert int(recording_row["dpf_at_acquisition"]) == 7
    stale_recording = registry.query_recording_detection_data_profile_latest(
        recording_ids=["recording_stale"]
    )
    assert stale_recording == []
    registry.close()


def test_query_keypoint_data_profile_latest_and_recording_latest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_shared",
        zarr_path=tmp_path / "a_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_shared",
        zarr_path=tmp_path / "b_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_a",
        recording_id="rec_shared",
        session_uuid="session_shared",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_b",
        recording_id="rec_shared",
        session_uuid="session_shared",
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
    assert dataset_latest[0]["genotype"] is None
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
    assert recording_latest[0]["genotype"] is None
    assert int(recording_latest[0]["dpf_at_acquisition"]) == 7
    registry.close()


def test_keypoint_profile_views_prefer_dataset_context_current(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_ctx",
        session_uuid="session_ctx",
        zarr_path=tmp_path / "keypoint_ctx_training.zarr",
        recording_id="recording_ctx",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_ctx",
        recording_id="recording_ctx",
        session_uuid="session_ctx",
    )
    _insert_keypoint_profile(
        registry,
        dataset_id="dataset_ctx",
        profile_run="kp_ctx",
        recording_id="recording_stale",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )

    dataset_latest = registry.query_keypoint_data_profile_latest(dataset_ids=["dataset_ctx"])
    assert len(dataset_latest) == 1
    row = dataset_latest[0]
    assert str(row["recording_id"]) == "recording_ctx"
    assert str(row["zarr_use"]) == "training"
    assert str(row["rig_id"]) == "rig_recording"
    assert str(row["camera_id"]) == "camera_recording"
    assert str(row["arena_id"]) == "arena_recording"
    assert str(row["dish_design"]) == "dish_design_recording"
    assert str(row["canvas_name"]) == "canvas_recording"
    assert str(row["protocol_name"]) == "protocol_recording"
    assert str(row["genotype"]) == "genotype_dataset_ctx"
    assert int(row["dpf_at_acquisition"]) == 7

    training_filtered = registry.query_keypoint_data_profile_latest(
        dataset_ids=["dataset_ctx"],
        zarr_use="training",
    )
    assert len(training_filtered) == 1
    analysis_filtered = registry.query_keypoint_data_profile_latest(
        dataset_ids=["dataset_ctx"],
        zarr_use="analysis",
    )
    assert analysis_filtered == []

    recording_latest = registry.query_recording_keypoint_data_profile_latest(
        recording_ids=["recording_ctx"]
    )
    assert len(recording_latest) == 1
    recording_row = recording_latest[0]
    assert str(recording_row["dataset_id"]) == "dataset_ctx"
    assert str(recording_row["recording_id"]) == "recording_ctx"
    assert str(recording_row["rig_id"]) == "rig_recording"
    assert str(recording_row["genotype"]) == "genotype_dataset_ctx"
    assert int(recording_row["dpf_at_acquisition"]) == 7
    stale_recording = registry.query_recording_keypoint_data_profile_latest(
        recording_ids=["recording_stale"]
    )
    assert stale_recording == []
    registry.close()


def test_query_eye_mask_data_profile_latest_and_recording_latest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_a",
        session_uuid="session_shared",
        zarr_path=tmp_path / "a_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "dataset_b",
        session_uuid="session_shared",
        zarr_path=tmp_path / "b_training.zarr",
        recording_id="rec_shared",
        zarr_use="training",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_a",
        recording_id="rec_shared",
        session_uuid="session_shared",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_b",
        recording_id="rec_shared",
        session_uuid="session_shared",
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
    assert dataset_latest[0]["genotype"] is None
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
    assert recording_latest[0]["genotype"] is None
    assert int(recording_latest[0]["dpf_at_acquisition"]) == 7
    registry.close()


def test_eye_mask_profile_views_prefer_dataset_context_current(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_ctx",
        session_uuid="session_ctx",
        zarr_path=tmp_path / "eye_mask_ctx_training.zarr",
        recording_id="recording_ctx",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_ctx",
        recording_id="recording_ctx",
        session_uuid="session_ctx",
    )
    _upsert_eye_mask_profile(
        registry,
        dataset_id="dataset_ctx",
        profile_run="eye_ctx",
        recording_id="recording_stale",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )

    dataset_latest = registry.query_eye_mask_data_profile_latest(dataset_ids=["dataset_ctx"])
    assert len(dataset_latest) == 1
    row = dataset_latest[0]
    assert str(row["recording_id"]) == "recording_ctx"
    assert str(row["zarr_use"]) == "training"
    assert str(row["rig_id"]) == "rig_recording"
    assert str(row["camera_id"]) == "camera_recording"
    assert str(row["arena_id"]) == "arena_recording"
    assert str(row["dish_design"]) == "dish_design_recording"
    assert str(row["canvas_name"]) == "canvas_recording"
    assert str(row["protocol_name"]) == "protocol_recording"
    assert str(row["genotype"]) == "genotype_dataset_ctx"
    assert int(row["dpf_at_acquisition"]) == 7

    training_filtered = registry.query_eye_mask_data_profile_latest(
        dataset_ids=["dataset_ctx"],
        zarr_use="training",
        stage_group="refined_eye_masks_runs",
    )
    assert len(training_filtered) == 1
    analysis_filtered = registry.query_eye_mask_data_profile_latest(
        dataset_ids=["dataset_ctx"],
        zarr_use="analysis",
        stage_group="refined_eye_masks_runs",
    )
    assert analysis_filtered == []

    recording_latest = registry.query_recording_eye_mask_data_profile_latest(
        recording_ids=["recording_ctx"],
        stage_group="refined_eye_masks_runs",
    )
    assert len(recording_latest) == 1
    recording_row = recording_latest[0]
    assert str(recording_row["dataset_id"]) == "dataset_ctx"
    assert str(recording_row["recording_id"]) == "recording_ctx"
    assert str(recording_row["rig_id"]) == "rig_recording"
    assert str(recording_row["genotype"]) == "genotype_dataset_ctx"
    assert int(recording_row["dpf_at_acquisition"]) == 7
    stale_recording = registry.query_recording_eye_mask_data_profile_latest(
        recording_ids=["recording_stale"],
        stage_group="refined_eye_masks_runs",
    )
    assert stale_recording == []
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
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_replace",
        recording_id="rec_replace",
        session_uuid="session_replace",
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
    assert str(latest[0]["genotype"]) == "genotype_dataset_replace"
    assert int(latest[0]["dpf_at_acquisition"]) == 7
    assert '"run":"c"' in str(latest[0]["profile_json"])
    registry.close()


def test_profile_upserts_keep_legacy_duplicate_context_without_canonical_owners(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_legacy",
        session_uuid="session_legacy",
        zarr_path=tmp_path / "legacy_training.zarr",
        recording_id="recording_legacy",
        zarr_use="training",
    )

    _upsert_profile(
        registry,
        dataset_id="dataset_legacy",
        profile_run="detect_legacy",
        recording_id="recording_profile",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )
    _insert_keypoint_profile(
        registry,
        dataset_id="dataset_legacy",
        profile_run="keypoint_legacy",
        recording_id="recording_profile",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )
    _upsert_eye_mask_profile(
        registry,
        dataset_id="dataset_legacy",
        profile_run="eye_legacy",
        recording_id="recording_profile",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )

    for table_name, profile_run in (
        ("detection_data_profile", "detect_legacy"),
        ("keypoint_data_profile", "keypoint_legacy"),
        ("eye_mask_data_profile", "eye_legacy"),
    ):
        row = _fetch_profile_row(
            registry,
            table_name=table_name,
            dataset_id="dataset_legacy",
            profile_run=profile_run,
        )
        _assert_profile_duplicate_context_values(
            row,
            rig_id="rig_profile",
            camera_id="camera_profile",
            arena_id="arena_profile",
            dish_design="dish_design_profile",
            canvas_name="canvas_profile",
            protocol_name="protocol_profile",
            genotype="genotype_profile",
            dpf_at_acquisition=12,
        )
    registry.close()


def test_profile_upserts_do_not_write_duplicate_context_when_canonical_owners_exist(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_ctx",
        session_uuid="session_ctx",
        zarr_path=tmp_path / "ctx_training.zarr",
        recording_id="recording_ctx",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_ctx",
        recording_id="recording_ctx",
        session_uuid="session_ctx",
    )

    _upsert_profile(
        registry,
        dataset_id="dataset_ctx",
        profile_run="detect_ctx_raw",
        recording_id="recording_stale",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )
    _insert_keypoint_profile(
        registry,
        dataset_id="dataset_ctx",
        profile_run="keypoint_ctx_raw",
        recording_id="recording_stale",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )
    _upsert_eye_mask_profile(
        registry,
        dataset_id="dataset_ctx",
        profile_run="eye_ctx_raw",
        recording_id="recording_stale",
        zarr_use="analysis",
        rig_id="rig_profile",
        camera_id="camera_profile",
        arena_id="arena_profile",
        dish_design="dish_design_profile",
        canvas_name="canvas_profile",
        protocol_name="protocol_profile",
        genotype="genotype_profile",
        dpf_at_acquisition=12,
    )

    for table_name, profile_run in (
        ("detection_data_profile", "detect_ctx_raw"),
        ("keypoint_data_profile", "keypoint_ctx_raw"),
        ("eye_mask_data_profile", "eye_ctx_raw"),
    ):
        row = _fetch_profile_row(
            registry,
            table_name=table_name,
            dataset_id="dataset_ctx",
            profile_run=profile_run,
        )
        _assert_profile_duplicate_context_cleared(row)
    registry.close()


def test_profile_upserts_freeze_existing_legacy_duplicate_context_after_canonical_backfill(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_freeze",
        session_uuid="session_freeze",
        zarr_path=tmp_path / "freeze_training.zarr",
        recording_id="recording_freeze",
        artifact_kind="source_recording",
        zarr_use="training",
    )

    _upsert_profile(
        registry,
        dataset_id="dataset_freeze",
        profile_run="detect_freeze",
        recording_id="recording_legacy",
        coverage_percent=70.0,
        rig_id="rig_legacy",
        camera_id="camera_legacy",
        arena_id="arena_legacy",
        dish_design="dish_design_legacy",
        canvas_name="canvas_legacy",
        protocol_name="protocol_legacy",
        genotype="genotype_legacy",
        dpf_at_acquisition=11,
    )
    _insert_keypoint_profile(
        registry,
        dataset_id="dataset_freeze",
        profile_run="keypoint_freeze",
        recording_id="recording_legacy",
        usable_rate=0.81,
        rig_id="rig_legacy",
        camera_id="camera_legacy",
        arena_id="arena_legacy",
        dish_design="dish_design_legacy",
        canvas_name="canvas_legacy",
        protocol_name="protocol_legacy",
        genotype="genotype_legacy",
        dpf_at_acquisition=11,
    )
    _upsert_eye_mask_profile(
        registry,
        dataset_id="dataset_freeze",
        profile_run="eye_freeze",
        recording_id="recording_legacy",
        usable_rate=0.82,
        rig_id="rig_legacy",
        camera_id="camera_legacy",
        arena_id="arena_legacy",
        dish_design="dish_design_legacy",
        canvas_name="canvas_legacy",
        protocol_name="protocol_legacy",
        genotype="genotype_legacy",
        dpf_at_acquisition=11,
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_freeze",
        recording_id="recording_freeze",
        session_uuid="session_freeze",
    )

    _upsert_profile(
        registry,
        dataset_id="dataset_freeze",
        profile_run="detect_freeze",
        recording_id="recording_new",
        coverage_percent=91.0,
        rig_id="rig_new",
        camera_id="camera_new",
        arena_id="arena_new",
        dish_design="dish_design_new",
        canvas_name="canvas_new",
        protocol_name="protocol_new",
        genotype="genotype_new",
        dpf_at_acquisition=13,
    )
    _insert_keypoint_profile(
        registry,
        dataset_id="dataset_freeze",
        profile_run="keypoint_freeze",
        recording_id="recording_new",
        usable_rate=0.93,
        rig_id="rig_new",
        camera_id="camera_new",
        arena_id="arena_new",
        dish_design="dish_design_new",
        canvas_name="canvas_new",
        protocol_name="protocol_new",
        genotype="genotype_new",
        dpf_at_acquisition=13,
    )
    _upsert_eye_mask_profile(
        registry,
        dataset_id="dataset_freeze",
        profile_run="eye_freeze",
        recording_id="recording_new",
        usable_rate=0.94,
        rig_id="rig_new",
        camera_id="camera_new",
        arena_id="arena_new",
        dish_design="dish_design_new",
        canvas_name="canvas_new",
        protocol_name="protocol_new",
        genotype="genotype_new",
        dpf_at_acquisition=13,
    )

    detection_row = _fetch_profile_row(
        registry,
        table_name="detection_data_profile",
        dataset_id="dataset_freeze",
        profile_run="detect_freeze",
    )
    _assert_profile_duplicate_context_values(
        detection_row,
        rig_id="rig_legacy",
        camera_id="camera_legacy",
        arena_id="arena_legacy",
        dish_design="dish_design_legacy",
        canvas_name="canvas_legacy",
        protocol_name="protocol_legacy",
        genotype="genotype_legacy",
        dpf_at_acquisition=11,
    )
    assert float(detection_row["coverage_percent"]) == 91.0

    keypoint_row = _fetch_profile_row(
        registry,
        table_name="keypoint_data_profile",
        dataset_id="dataset_freeze",
        profile_run="keypoint_freeze",
    )
    _assert_profile_duplicate_context_values(
        keypoint_row,
        rig_id="rig_legacy",
        camera_id="camera_legacy",
        arena_id="arena_legacy",
        dish_design="dish_design_legacy",
        canvas_name="canvas_legacy",
        protocol_name="protocol_legacy",
        genotype="genotype_legacy",
        dpf_at_acquisition=11,
    )
    assert float(keypoint_row["usable_rate"]) == 0.93

    eye_row = _fetch_profile_row(
        registry,
        table_name="eye_mask_data_profile",
        dataset_id="dataset_freeze",
        profile_run="eye_freeze",
    )
    _assert_profile_duplicate_context_values(
        eye_row,
        rig_id="rig_legacy",
        camera_id="camera_legacy",
        arena_id="arena_legacy",
        dish_design="dish_design_legacy",
        canvas_name="canvas_legacy",
        protocol_name="protocol_legacy",
        genotype="genotype_legacy",
        dpf_at_acquisition=11,
    )
    assert float(eye_row["usable_rate"]) == 0.94
    registry.close()


def test_profile_replace_stops_writing_duplicate_context_when_canonical_owners_exist(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset(
        "dataset_replace_ctx",
        session_uuid="session_replace_ctx",
        zarr_path=tmp_path / "replace_ctx_training.zarr",
        recording_id="recording_replace_ctx",
        artifact_kind="source_recording",
        zarr_use="training",
    )
    _seed_canonical_context_for_dataset(
        registry,
        root=tmp_path,
        dataset_id="dataset_replace_ctx",
        recording_id="recording_replace_ctx",
        session_uuid="session_replace_ctx",
    )

    registry.replace_detection_data_profile(
        "dataset_replace_ctx",
        [
            {
                "profile_run": "detect_replace",
                "recording_id": "recording_stale",
                "zarr_use": "analysis",
                "detection_type": "manual",
                "detection_path": "refined_detect_runs/refined_replace/manual",
                "profile_created_utc": "2026-02-24T00:00:00+00:00",
                "frames_total": 100,
                "frames_with_detections": 90,
                "coverage_percent": 90.0,
                "detections_total": 100,
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
                "edge_proximity_rate": 0.05,
                "rig_id": "rig_profile",
                "camera_id": "camera_profile",
                "arena_id": "arena_profile",
                "dish_design": "dish_design_profile",
                "canvas_name": "canvas_profile",
                "protocol_name": "protocol_profile",
                "genotype": "genotype_profile",
                "dpf_at_acquisition": 12,
                "profile_json": '{"schema_name":"detection_dataset_profile","schema_version":"v1"}',
                "zarr_mtime_ns": 123,
            }
        ],
    )
    registry.replace_keypoint_data_profile(
        "dataset_replace_ctx",
        [
            {
                "profile_run": "keypoint_replace",
                "recording_id": "recording_stale",
                "zarr_use": "analysis",
                "keypoint_method": "traditional_pose",
                "source_keypoint_path": "refined_keypoint_runs/keypoint_replace/traditional_pose",
                "source_keypoint_run": "keypoint_replace",
                "skeleton_id": "fish_v1",
                "kpt_shape": "[3,3]",
                "profile_created_utc": "2026-02-24T00:00:00+00:00",
                "rows_total": 200,
                "rows_usable": 180,
                "usable_keypoints_total": 540,
                "usable_rate": 0.90,
                "confidence_valid_rate": 0.95,
                "geometry_valid_rate": 0.96,
                "triangle_area_p10": 0.01,
                "triangle_area_p50": 0.02,
                "triangle_area_p90": 0.03,
                "min_angle_p10": 10.0,
                "min_angle_p50": 20.0,
                "min_angle_p90": 30.0,
                "heading_p10": -0.4,
                "heading_p50": 0.0,
                "heading_p90": 0.4,
                "rig_id": "rig_profile",
                "camera_id": "camera_profile",
                "arena_id": "arena_profile",
                "dish_design": "dish_design_profile",
                "canvas_name": "canvas_profile",
                "protocol_name": "protocol_profile",
                "genotype": "genotype_profile",
                "dpf_at_acquisition": 12,
                "profile_json": '{"schema_name":"keypoint_dataset_profile","schema_version":"v1"}',
                "zarr_mtime_ns": 123,
            }
        ],
    )
    registry.replace_eye_mask_data_profile(
        "dataset_replace_ctx",
        [
            {
                "profile_run": "eye_replace",
                "recording_id": "recording_stale",
                "zarr_use": "analysis",
                "stage_group": "refined_eye_masks_runs",
                "eye_mask_method": "traditional",
                "source_eye_mask_path": "refined_eye_masks_runs/eye_replace",
                "source_eye_mask_run": "eye_replace",
                "source_keypoint_path": "refined_keypoints_runs/refined_keypoints_001",
                "source_keypoint_run": "refined_keypoints_001",
                "source_crop_run": "crop_001",
                "profile_created_utc": "2026-02-24T00:00:00+00:00",
                "rows_total": 200,
                "rows_usable": 180,
                "usable_rate": 0.90,
                "reviewed_rate": 1.0,
                "excluded_rate": 0.10,
                "exclusion_reasons_json": '{"ellipse_fit_failed":20}',
                "ellipse_success_rate": 0.95,
                "pair_success_rate": 0.90,
                "area_p10": 300.0,
                "area_p50": 420.0,
                "area_p90": 560.0,
                "left_area_p10": 145.0,
                "left_area_p50": 205.0,
                "left_area_p90": 265.0,
                "right_area_p10": 150.0,
                "right_area_p50": 210.0,
                "right_area_p90": 275.0,
                "union_area_p10": 300.0,
                "union_area_p50": 415.0,
                "union_area_p90": 545.0,
                "area_lr_ratio_p10": 0.90,
                "area_lr_ratio_p50": 0.98,
                "area_lr_ratio_p90": 1.08,
                "major_axis_p10": 14.0,
                "major_axis_p50": 19.0,
                "major_axis_p90": 25.0,
                "minor_axis_p10": 7.0,
                "minor_axis_p50": 10.0,
                "minor_axis_p90": 13.0,
                "aspect_ratio_p10": 1.4,
                "aspect_ratio_p50": 1.8,
                "aspect_ratio_p90": 2.2,
                "eye_separation_p10": 21.0,
                "eye_separation_p50": 27.0,
                "eye_separation_p90": 34.0,
                "edge_proximity_rate": 0.05,
                "review_state": "approved",
                "review_method": "manual",
                "review_intended_use": "training",
                "review_timestamp_utc": "2026-02-24T00:05:00+00:00",
                "source_keypoint_stale_state": "fresh",
                "source_keypoint_stale_reason": None,
                "source_keypoint_stale_timestamp_utc": "2026-02-24T00:04:00+00:00",
                "source_keypoint_stale_json": '{"state":"fresh"}',
                "rig_id": "rig_profile",
                "camera_id": "camera_profile",
                "arena_id": "arena_profile",
                "dish_design": "dish_design_profile",
                "canvas_name": "canvas_profile",
                "protocol_name": "protocol_profile",
                "genotype": "genotype_profile",
                "dpf_at_acquisition": 12,
                "profile_json": '{"schema_name":"eye_mask_dataset_profile","schema_version":"v1"}',
                "zarr_mtime_ns": 123,
            }
        ],
    )

    for table_name, profile_run in (
        ("detection_data_profile", "detect_replace"),
        ("keypoint_data_profile", "keypoint_replace"),
        ("eye_mask_data_profile", "eye_replace"),
    ):
        row = _fetch_profile_row(
            registry,
            table_name=table_name,
            dataset_id="dataset_replace_ctx",
            profile_run=profile_run,
        )
        _assert_profile_duplicate_context_cleared(row)
    registry.close()
