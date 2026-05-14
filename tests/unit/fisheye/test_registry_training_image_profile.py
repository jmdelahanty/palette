from __future__ import annotations

from pathlib import Path

from fisheye.registry.db import Registry


def _seed_training_dataset(registry: Registry, tmp_path: Path) -> None:
    registry.upsert_recording(
        recording_id="rec_training_img",
        session_uuid="session_training_img",
        recording_name="sleepyfish_001",
        recording_path=str(tmp_path / "sleepyfish_001"),
        recording_type="recording_only",
        rig_id="omnifin0",
        arena_id="arena_1",
        camera_id="2010095",
        canvas_name="sleepyfish",
        protocol_name="recording_only",
        dish_design="palm",
    )
    registry.upsert_dataset(
        "dataset_training_img",
        session_uuid="session_training_img",
        zarr_path=tmp_path / "sleepyfish_001_training.zarr",
        recording_id="rec_training_img",
        artifact_kind="source_recording",
        zarr_use="training",
    )


def test_schema_has_training_image_profile_table_view_and_indexes(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        table = registry.conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'training_image_profile';
            """
        ).fetchone()
        view = registry.conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'view' AND name = 'training_image_profile_latest';
            """
        ).fetchone()
        indexes = {
            row["name"]
            for row in registry.conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'index' AND tbl_name = 'training_image_profile';
                """
            ).fetchall()
        }
    finally:
        registry.close()

    assert table is not None
    assert view is not None
    assert "idx_training_image_profile_recording_created" in indexes
    assert "idx_training_image_profile_scope" in indexes
    assert "idx_training_image_profile_domain_metrics" in indexes


def test_upsert_and_query_training_image_profile_latest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        _seed_training_dataset(registry, tmp_path)
        registry.upsert_training_image_profile(
            dataset_id="dataset_training_img",
            profile_run="training_image_profile_old",
            recording_id="rec_training_img",
            zarr_use="training",
            source_frame_array="raw_video/images_ds",
            profile_created_utc="2026-05-13T10:00:00+00:00",
            frames_total=100,
            frames_profiled=50,
            mean_intensity_p50=100.0,
            contrast_p50=20.0,
            sharpness_p50=5.0,
            clip_dark_rate_mean=0.01,
            clip_bright_rate_mean=0.02,
            illumination_center_edge_p50=3.0,
            illumination_slope_x_p50=1.0,
            illumination_slope_y_p50=-1.0,
            fish_bg_contrast_p50=7.0,
            rig_id="legacy_rig",
            camera_id="legacy_camera",
            arena_id="legacy_arena",
            dish_design="legacy_dish",
            canvas_name="legacy_canvas",
            protocol_name="legacy_protocol",
            genotype="legacy_genotype",
            dpf_at_acquisition=5,
            profile_json='{"schema_name":"training_image_profile","schema_version":"v1","run":"old"}',
        )
        registry.upsert_training_image_profile(
            dataset_id="dataset_training_img",
            profile_run="training_image_profile_new",
            recording_id="rec_training_img",
            zarr_use="training",
            source_frame_array="raw_video/images_ds",
            profile_created_utc="2026-05-13T11:00:00+00:00",
            frames_total=100,
            frames_profiled=100,
            mean_intensity_p50=110.0,
            contrast_p50=25.0,
            sharpness_p50=8.0,
            clip_dark_rate_mean=0.03,
            clip_bright_rate_mean=0.04,
            illumination_center_edge_p50=2.0,
            illumination_slope_x_p50=0.5,
            illumination_slope_y_p50=-0.5,
            fish_bg_contrast_p50=9.0,
            rig_id="legacy_rig",
            camera_id="legacy_camera",
            arena_id="legacy_arena",
            dish_design="legacy_dish",
            canvas_name="legacy_canvas",
            protocol_name="legacy_protocol",
            genotype="legacy_genotype",
            dpf_at_acquisition=5,
            profile_json='{"schema_name":"training_image_profile","schema_version":"v1","run":"new"}',
        )

        rows = registry.query_training_image_profile_latest(
            dataset_ids=["dataset_training_img"],
            zarr_use="training",
            min_frames_profiled=80,
        )
    finally:
        registry.close()

    assert len(rows) == 1
    row = dict(rows[0])
    assert row["profile_run"] == "training_image_profile_new"
    assert row["mean_intensity_p50"] == 110.0
    assert row["contrast_p50"] == 25.0
    assert row["recording_id"] == "rec_training_img"
    assert row["zarr_use"] == "training"
    assert row["rig_id"] == "omnifin0"
    assert row["camera_id"] == "2010095"
    assert row["dish_design"] == "palm"
