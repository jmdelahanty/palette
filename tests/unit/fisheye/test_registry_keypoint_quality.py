"""Unit tests for keypoint quality registry schema and query filtering."""

from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry


def _create_pose_with_quality(
    path: Path,
    *,
    session_uuid: str,
    method: str,
    review_state: str,
    intended_use: str,
    usable_rows: int,
    total_rows: int = 4,
) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["session_uuid"] = session_uuid
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.zeros((total_rows, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))
    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_001")
    crop.attrs["detection_source_type"] = "filtered"
    crop.create_array("roi_images", data=np.zeros((total_rows, 8, 8), dtype=np.uint8), chunks=(1, 8, 8))
    kp_parent = root.create_group("keypoints_runs")
    kp = kp_parent.create_group("kp_001")
    kp.attrs["method"] = method
    kp.attrs["source_crop_run"] = "crop_001"
    kp.create_array("keypoints_roi", data=np.zeros((total_rows, 3, 2), dtype=np.float32), chunks=(1, 3, 2))
    refined_parent = root.create_group("refined_keypoints_runs")
    refined = refined_parent.create_group("refined_001")
    refined.attrs["source_keypoints_run"] = "kp_001"
    refined.attrs["created_utc"] = "2026-02-08T00:00:00+00:00"
    refined.attrs["keypoint_review_status"] = {
        "state": review_state,
        "intended_use": intended_use,
        "timestamp_utc": "2026-02-08T00:00:00+00:00",
    }
    refined.create_array(
        "usable_keypoints",
        data=np.array([True] * usable_rows + [False] * (total_rows - usable_rows), dtype=np.bool_),
        chunks=(total_rows,),
    )


def test_query_keypoint_quality_current_filters_by_review_and_rate(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    a_path = tmp_path / "a.zarr"
    b_path = tmp_path / "b.zarr"
    _create_pose_with_quality(
        a_path,
        session_uuid="session_a",
        method="traditional_pose",
        review_state="approved",
        intended_use="training",
        usable_rows=3,
    )
    _create_pose_with_quality(
        b_path,
        session_uuid="session_b",
        method="traditional_pose",
        review_state="pending",
        intended_use="training",
        usable_rows=1,
    )
    registry.register_from_root(zarr.open_group(str(a_path), mode="r"), a_path)
    registry.register_from_root(zarr.open_group(str(b_path), mode="r"), b_path)

    rows = registry.query_keypoint_quality_current(
        review_state="approved",
        review_intended_use="training",
        min_usable_keypoints_rate=0.7,
        keypoint_method="traditional_pose",
    )
    ids = {str(row["dataset_id"]) for row in rows}
    assert ids == {"session_a"}
    registry.close()


def test_keypoint_quality_current_view_keeps_latest_per_dataset_method(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_x", session_uuid="dataset_x", zarr_path=tmp_path / "x.zarr")
    registry.upsert_keypoint_quality(
        dataset_id="dataset_x",
        refined_run="refined_old",
        refined_created_utc="2026-02-07T00:00:00+00:00",
        source_keypoint_run="kp_old",
        keypoint_method="traditional_pose",
        review_state="approved",
        review_intended_use="training",
        review_reviewer=None,
        review_timestamp_utc="2026-02-07T00:00:00+00:00",
        usable_keypoints=3,
        total_keypoints=4,
        usable_keypoints_rate=0.75,
        raw_keypoints_success_rate=0.8,
        raw_keypoints_successful=3,
    )
    registry.upsert_keypoint_quality(
        dataset_id="dataset_x",
        refined_run="refined_new",
        refined_created_utc="2026-02-08T00:00:00+00:00",
        source_keypoint_run="kp_new",
        keypoint_method="traditional_pose",
        review_state="approved",
        review_intended_use="training",
        review_reviewer=None,
        review_timestamp_utc="2026-02-08T00:00:00+00:00",
        usable_keypoints=4,
        total_keypoints=4,
        usable_keypoints_rate=1.0,
        raw_keypoints_success_rate=1.0,
        raw_keypoints_successful=4,
    )
    row = registry.query_keypoint_quality_current(
        dataset_ids=["dataset_x"],
        keypoint_method="traditional_pose",
    )[0]
    assert str(row["refined_run"]) == "refined_new"
    assert str(row["source_keypoint_run"]) == "kp_new"
    registry.close()


def test_keypoint_quality_overview_view_exposes_expected_columns(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_x", session_uuid="dataset_x", zarr_path=tmp_path / "x.zarr")
    registry.upsert_keypoint_quality(
        dataset_id="dataset_x",
        refined_run="refined_new",
        refined_created_utc="2026-02-08T00:00:00+00:00",
        source_keypoint_run="kp_new",
        keypoint_method="traditional_pose",
        review_state="approved",
        review_intended_use="training",
        review_reviewer="tester",
        review_timestamp_utc="2026-02-08T00:00:00+00:00",
        usable_keypoints=4,
        total_keypoints=4,
        usable_keypoints_rate=1.0,
        raw_keypoints_success_rate=1.0,
        raw_keypoints_successful=4,
    )
    row = registry.conn.execute(
        "SELECT * FROM keypoint_quality_overview WHERE dataset_id = ?;",
        ("dataset_x",),
    ).fetchone()
    assert row is not None
    expected_columns = {
        "dataset_id",
        "zarr_path",
        "zarr_purpose",
        "keypoint_method",
        "source_keypoint_run",
        "refined_run",
        "review_state",
        "review_intended_use",
        "usable_keypoints",
        "total_keypoints",
        "usable_keypoints_rate",
        "quality_updated_utc",
        "zarr_mtime_ns",
        "quality_stale",
    }
    assert expected_columns.issubset(set(row.keys()))
    registry.close()
