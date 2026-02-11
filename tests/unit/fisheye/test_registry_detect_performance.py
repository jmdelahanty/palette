"""Unit tests for detect performance registry extraction and views."""

from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry


def _create_detect_archive(path: Path, *, session_uuid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs["latest"] = "detect_new"

    detect_old = detect_parent.create_group("detect_old")
    detect_old.attrs["detect_timestamp_utc"] = "2026-02-08T00:00:00+00:00"
    detect_old.attrs["detection_method"] = "yolo"
    detect_old.attrs["model_path"] = "/tmp/models/detect_old.pt"
    detect_old.attrs["model_name"] = "detect_old.pt"
    detect_old.attrs["summary_statistics"] = {
        "percent_frames_with_detections": 50.0,
        "frames_with_detections": 2,
        "frames_zero_detections": 2,
        "total_frames": 4,
        "mean_confidence": 0.82,
    }
    detect_old.attrs["inference_average_fps"] = 70.0
    detect_old.create_array("frame_counts", data=np.array([1, 1, 0, 0], dtype=np.int32), chunks=(4,))

    detect_new = detect_parent.create_group("detect_new")
    detect_new.attrs["detect_timestamp_utc"] = "2026-02-09T00:00:00+00:00"
    detect_new.attrs["provenance"] = {"method": "yolo"}
    detect_new.attrs["model_resolution_selected_run_id"] = "omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb"
    detect_new.attrs["model_resolution_selected_set_id"] = "detect_cedar_shadow_v007"
    detect_new.attrs["inference_average_fps"] = 90.0
    detect_new.create_array("frame_counts", data=np.array([1, 1, 0, 1], dtype=np.int32), chunks=(4,))


def _create_detectless_archive(path: Path, *, session_uuid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((4, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))


def test_register_from_root_populates_detect_performance_all_runs_and_latest(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_a" / "zarr" / "rec_a_analysis.zarr"
    _create_detect_archive(zarr_path, session_uuid="rec_a_uuid")

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)

    count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM detect_performance WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert count is not None
    assert int(count["n"]) == 2

    latest = registry.conn.execute(
        """
        SELECT detect_run, coverage_percent, detection_method, inference_average_fps, model_run_id, model_set_id
        FROM detect_performance_latest
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert latest is not None
    assert str(latest["detect_run"]) == "detect_new"
    assert float(latest["coverage_percent"]) == 75.0
    assert str(latest["detection_method"]) == "yolo"
    assert float(latest["inference_average_fps"]) == 90.0
    assert str(latest["model_run_id"]) == "omnifin0_cedar_shadow_v007_detect_20260206-235656_25f3fbcb"
    assert str(latest["model_set_id"]) == "detect_cedar_shadow_v007"

    rec_latest = registry.conn.execute(
        """
        SELECT recording_id, dataset_id, detect_run
        FROM recording_detect_performance_latest
        WHERE recording_id = ?;
        """,
        ("rec_a_uuid",),
    ).fetchone()
    assert rec_latest is not None
    assert str(rec_latest["dataset_id"]) == dataset_id
    assert str(rec_latest["detect_run"]) == "detect_new"
    registry.close()


def test_register_from_root_handles_archive_without_detect_runs(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_no_detect" / "zarr" / "rec_no_detect_analysis.zarr"
    _create_detectless_archive(zarr_path, session_uuid="rec_no_detect_uuid")

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)
    count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM detect_performance WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert count is not None
    assert int(count["n"]) == 0
    registry.close()


def test_model_only_views_select_latest_model_backed_run(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_b" / "zarr" / "rec_b_analysis.zarr"
    _create_detect_archive(zarr_path, session_uuid="rec_b_uuid")

    dataset_id = registry.register_from_root(zarr.open_group(str(zarr_path), mode="r"), zarr_path)

    latest_any = registry.conn.execute(
        "SELECT detect_run FROM detect_performance_latest WHERE dataset_id = ?;",
        (dataset_id,),
    ).fetchone()
    assert latest_any is not None
    assert str(latest_any["detect_run"]) == "detect_new"

    latest_model = registry.conn.execute(
        """
        SELECT detect_run, model_name
        FROM detect_model_performance_latest
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert latest_model is not None
    assert str(latest_model["detect_run"]) == "detect_old"
    assert str(latest_model["model_name"]) == "detect_old.pt"

    rec_latest_model = registry.conn.execute(
        """
        SELECT recording_id, detect_run
        FROM recording_detect_model_performance_latest
        WHERE recording_id = ?;
        """,
        ("rec_b_uuid",),
    ).fetchone()
    assert rec_latest_model is not None
    assert str(rec_latest_model["detect_run"]) == "detect_old"
    registry.close()


def test_model_summary_views_aggregate_counts_and_percentiles(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    dataset_rows = [
        ("dataset_a", "rec_a", "a.zarr"),
        ("dataset_b", "rec_b", "b.zarr"),
        ("dataset_c", "rec_c", "c.zarr"),
    ]
    for dataset_id, recording_id, filename in dataset_rows:
        registry.upsert_dataset(
            dataset_id,
            session_uuid=recording_id,
            zarr_path=tmp_path / filename,
            recording_id=recording_id,
            artifact_kind="source_recording",
            zarr_use="analysis",
        )

    registry.upsert_detect_performance(
        dataset_id="dataset_a",
        detect_run="detect_a",
        detect_created_utc="2026-02-09T10:00:00+00:00",
        recording_id="rec_a",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id="run_x",
        model_set_id="set_x",
        model_path="/models/x.pt",
        model_name="x.pt",
        coverage_percent=80.0,
        frames_with_detections=80,
        frames_zero_detections=20,
        total_frames=100,
        mean_confidence=0.8,
        min_confidence=0.4,
        max_confidence=1.0,
        inference_duration_seconds=10.0,
        inference_average_fps=100.0,
        inference_avg_batch_ms=40.0,
        inference_avg_read_ms=100.0,
        conf_threshold=0.4,
        iou_threshold=0.8,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )
    registry.upsert_detect_performance(
        dataset_id="dataset_b",
        detect_run="detect_b",
        detect_created_utc="2026-02-09T10:01:00+00:00",
        recording_id="rec_b",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id="run_x",
        model_set_id="set_x",
        model_path="/models/x.pt",
        model_name="x.pt",
        coverage_percent=90.0,
        frames_with_detections=90,
        frames_zero_detections=10,
        total_frames=100,
        mean_confidence=0.9,
        min_confidence=0.5,
        max_confidence=1.0,
        inference_duration_seconds=10.0,
        inference_average_fps=80.0,
        inference_avg_batch_ms=50.0,
        inference_avg_read_ms=120.0,
        conf_threshold=0.4,
        iou_threshold=0.8,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )
    registry.upsert_detect_performance(
        dataset_id="dataset_c",
        detect_run="detect_c",
        detect_created_utc="2026-02-09T10:02:00+00:00",
        recording_id="rec_c",
        zarr_use="analysis",
        detection_method="yolo",
        model_run_id="run_y",
        model_set_id="set_y",
        model_path="/models/y.pt",
        model_name="y.pt",
        coverage_percent=70.0,
        frames_with_detections=70,
        frames_zero_detections=30,
        total_frames=100,
        mean_confidence=0.75,
        min_confidence=0.35,
        max_confidence=1.0,
        inference_duration_seconds=10.0,
        inference_average_fps=60.0,
        inference_avg_batch_ms=60.0,
        inference_avg_read_ms=130.0,
        conf_threshold=0.4,
        iou_threshold=0.8,
        batch_size=16,
        inference_width=640,
        inference_height=640,
    )

    dataset_summary = registry.conn.execute(
        """
        SELECT
            model_run_id,
            model_set_id,
            dataset_count,
            recording_count,
            coverage_avg,
            coverage_p10,
            coverage_p50,
            coverage_p90,
            fps_avg,
            fps_p10,
            fps_p50,
            fps_p90,
            read_ms_avg,
            read_ms_p10,
            read_ms_p50,
            read_ms_p90
        FROM detect_model_performance_summary
        WHERE model_run_id = 'run_x';
        """
    ).fetchone()
    assert dataset_summary is not None
    assert int(dataset_summary["dataset_count"]) == 2
    assert int(dataset_summary["recording_count"]) == 2
    assert float(dataset_summary["coverage_avg"]) == 85.0
    assert float(dataset_summary["coverage_p10"]) == 80.0
    assert float(dataset_summary["coverage_p50"]) == 80.0
    assert float(dataset_summary["coverage_p90"]) == 90.0
    assert float(dataset_summary["fps_avg"]) == 90.0
    assert float(dataset_summary["fps_p10"]) == 80.0
    assert float(dataset_summary["fps_p50"]) == 80.0
    assert float(dataset_summary["fps_p90"]) == 100.0
    assert float(dataset_summary["read_ms_avg"]) == 110.0
    assert float(dataset_summary["read_ms_p10"]) == 100.0
    assert float(dataset_summary["read_ms_p50"]) == 100.0
    assert float(dataset_summary["read_ms_p90"]) == 120.0

    recording_summary = registry.conn.execute(
        """
        SELECT dataset_count, recording_count
        FROM recording_detect_model_performance_summary
        WHERE model_run_id = 'run_x';
        """
    ).fetchone()
    assert recording_summary is not None
    assert int(recording_summary["dataset_count"]) == 2
    assert int(recording_summary["recording_count"]) == 2
    registry.close()
