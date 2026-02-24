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
        profile_json='{"schema_name":"detection_dataset_profile","schema_version":"v1"}',
        zarr_mtime_ns=123,
        updated_utc=updated_utc,
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
    assert '"run":"c"' in str(latest[0]["profile_json"])
    registry.close()
