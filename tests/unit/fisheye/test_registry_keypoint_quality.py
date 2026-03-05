"""Unit tests for keypoint quality registry schema and query filtering."""

from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry, _extract_keypoint_quality_rows


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, key):
        return self._data[key]


class _FakeGroup:
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        self.attrs: dict[str, object] = dict(attrs or {})
        self._children: dict[str, object] = {}

    def add_group(self, name: str, *, attrs: dict[str, object] | None = None) -> "_FakeGroup":
        group = _FakeGroup(attrs=attrs)
        self._children[name] = group
        return group

    def add_array(self, name: str, data: np.ndarray) -> _FakeArray:
        arr = _FakeArray(np.asarray(data))
        self._children[name] = arr
        return arr

    def get(self, key: str):
        return self._children.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str):
        return self._children[key]

    def keys(self):
        return self._children.keys()

    def group_keys(self):
        return [name for name, value in self._children.items() if isinstance(value, _FakeGroup)]


def _create_pose_with_quality(
    path: Path,
    *,
    session_uuid: str,
    method: str,
    review_state: str,
    intended_use: str,
    usable_rows: int,
    total_rows: int = 4,
    review_method: str = "manual",
    review_reviewer: str = "pytest",
    review_notes: str = "looks good",
    review_timestamp_key: str = "timestamp_utc",
    review_policy_id: str | None = None,
    review_policy_version: int | None = None,
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
    review_status = {
        "state": review_state,
        "method": review_method,
        "intended_use": intended_use,
        "reviewer": review_reviewer,
        "notes": review_notes,
    }
    if review_policy_id is not None or review_policy_version is not None:
        review_status["auto_review"] = {
            "policy_id": review_policy_id,
            "policy_version": review_policy_version,
        }
    review_status[review_timestamp_key] = "2026-02-08T00:00:00+00:00"
    refined.attrs["keypoint_review_status"] = review_status
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
        review_method="manual",
        review_notes="old",
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
        review_method="hybrid",
        review_notes="new",
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
    assert str(row["review_method"]) == "hybrid"
    assert str(row["review_notes"]) == "new"
    registry.close()


def test_keypoint_quality_current_view_prefers_latest_keypoint_source_run(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_x", session_uuid="dataset_x", zarr_path=tmp_path / "x.zarr")
    registry.upsert_keypoint_performance(
        dataset_id="dataset_x",
        keypoint_run="kp_latest",
        keypoint_created_utc="2026-02-10T00:00:00+00:00",
        recording_id=None,
        zarr_use=None,
        keypoint_method="yolo_pose",
        model_run_id=None,
        model_set_id=None,
        model_path=None,
        model_name=None,
        source_crop_run=None,
        source_detect_run=None,
        source_refined_run=None,
        total_rois=4,
        successful_detections=4,
        failed_detections=0,
        success_rate_percent=100.0,
        frames_with_keypoints=None,
        mean_confidence=None,
        duration_seconds=None,
        inference_duration_seconds=None,
        keypoints_per_second=None,
        inference_average_fps=None,
        batch_size=None,
        imgsz=None,
        conf_threshold=None,
        iou_threshold=None,
        summary_statistics_json=None,
    )
    registry.upsert_keypoint_quality(
        dataset_id="dataset_x",
        refined_run="refined_stale",
        refined_created_utc="2026-02-08T00:00:00+00:00",
        source_keypoint_run="kp_old",
        keypoint_method="yolo_pose",
        review_state="approved",
        review_intended_use="training",
        review_reviewer=None,
        review_timestamp_utc="2026-02-11T00:00:00+00:00",
        review_method="manual",
        review_notes="reviewed later",
        usable_keypoints=4,
        total_keypoints=4,
        usable_keypoints_rate=1.0,
        raw_keypoints_success_rate=1.0,
        raw_keypoints_successful=4,
    )
    registry.upsert_keypoint_quality(
        dataset_id="dataset_x",
        refined_run="refined_latest_source",
        refined_created_utc="2026-02-10T00:00:00+00:00",
        source_keypoint_run="kp_latest",
        keypoint_method="yolo_pose",
        review_state="approved",
        review_intended_use="training",
        review_reviewer=None,
        review_timestamp_utc="2026-02-09T00:00:00+00:00",
        review_method="manual",
        review_notes="matches latest keypoint run",
        usable_keypoints=4,
        total_keypoints=4,
        usable_keypoints_rate=1.0,
        raw_keypoints_success_rate=1.0,
        raw_keypoints_successful=4,
    )

    row = registry.query_keypoint_quality_current(
        dataset_ids=["dataset_x"],
        keypoint_method="yolo_pose",
    )[0]
    assert str(row["refined_run"]) == "refined_latest_source"
    assert str(row["source_keypoint_run"]) == "kp_latest"
    registry.close()


def test_quality_tables_have_aligned_shared_review_columns(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    shared = {
        "review_state",
        "review_method",
        "review_intended_use",
        "review_reviewer",
        "review_notes",
        "review_timestamp_utc",
    }
    detect_cols = {str(row["name"]) for row in registry.conn.execute("PRAGMA table_info(detect_quality);").fetchall()}
    keypoint_cols = {
        str(row["name"]) for row in registry.conn.execute("PRAGMA table_info(keypoint_quality);").fetchall()
    }
    assert shared.issubset(detect_cols)
    assert shared.issubset(keypoint_cols)
    assert "review_policy_id" in keypoint_cols
    assert "review_policy_version" in keypoint_cols
    assert "review_resolved_group" in detect_cols
    registry.close()


def test_keypoint_quality_extracts_shared_review_fields_and_legacy_timestamp_alias(tmp_path: Path) -> None:
    root = _FakeGroup()
    keypoints_parent = root.add_group("keypoints_runs")
    keypoint_run = keypoints_parent.add_group("kp_001", attrs={"method": "traditional_pose"})
    keypoint_run.add_array("keypoints_roi", np.zeros((4, 3, 2), dtype=np.float32))

    refined_parent = root.add_group("refined_keypoints_runs")
    refined = refined_parent.add_group(
        "refined_001",
        attrs={
            "source_keypoints_run": "kp_001",
            "created_utc": "2026-02-08T00:00:00+00:00",
            "keypoint_review_status": {
                "state": "approved",
                "method": "spotcheck",
                "intended_use": "training",
                "reviewer": "reviewer_a",
                "notes": "legacy alias",
                "reviewed_at": "2026-02-08T00:00:00+00:00",
                "auto_review": {
                    "policy_id": "keypoint_auto_review_v1",
                    "policy_version": 1,
                },
            },
        },
    )
    refined.add_array("usable_keypoints", np.array([True, True, True, False], dtype=np.bool_))

    rows = _extract_keypoint_quality_rows(root, zarr_path=tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["review_state"] == "approved"
    assert row["review_method"] == "spotcheck"
    assert row["review_intended_use"] == "training"
    assert row["review_reviewer"] == "reviewer_a"
    assert row["review_notes"] == "legacy alias"
    assert row["review_policy_id"] == "keypoint_auto_review_v1"
    assert row["review_policy_version"] == 1
    assert row["review_timestamp_utc"] == "2026-02-08T00:00:00+00:00"


def test_replace_keypoint_quality_auto_adds_review_columns_for_legacy_table(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_keypoint", session_uuid="dataset_keypoint", zarr_path=tmp_path / "kp.zarr")
    registry.conn.execute("DROP VIEW IF EXISTS keypoint_quality_current;")
    registry.conn.execute("DROP TABLE keypoint_quality;")
    registry.conn.execute(
        """
        CREATE TABLE keypoint_quality (
            dataset_id TEXT NOT NULL,
            refined_run TEXT NOT NULL,
            refined_created_utc TEXT,
            source_keypoint_run TEXT NOT NULL,
            keypoint_method TEXT,
            review_state TEXT,
            review_intended_use TEXT,
            review_reviewer TEXT,
            review_timestamp_utc TEXT,
            usable_keypoints INTEGER,
            total_keypoints INTEGER,
            usable_keypoints_rate REAL,
            raw_keypoints_success_rate REAL,
            raw_keypoints_successful INTEGER,
            quality_updated_utc TEXT,
            zarr_mtime_ns INTEGER,
            PRIMARY KEY (dataset_id, refined_run),
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );
        """
    )
    registry.conn.commit()

    registry.replace_keypoint_quality(
        "dataset_keypoint",
        [
            {
                "refined_run": "refined_legacy",
                "refined_created_utc": "2026-02-10T00:00:00+00:00",
                "source_keypoint_run": "kp_001",
                "keypoint_method": "traditional_pose",
                "review_state": "approved",
                "review_method": "manual",
                "review_intended_use": "training",
                "review_reviewer": "reviewer_keypoint",
                "review_notes": "added via ensure_columns",
                "review_policy_id": "keypoint_auto_review_v1",
                "review_policy_version": 1,
                "review_timestamp_utc": "2026-02-10T00:01:00+00:00",
                "usable_keypoints": 3,
                "total_keypoints": 4,
                "usable_keypoints_rate": 0.75,
                "raw_keypoints_success_rate": 0.80,
                "raw_keypoints_successful": 3,
            }
        ],
    )

    cols = {
        str(row["name"])
        for row in registry.conn.execute("PRAGMA table_info(keypoint_quality);").fetchall()
    }
    assert "review_method" in cols
    assert "review_notes" in cols
    assert "review_policy_id" in cols
    assert "review_policy_version" in cols
    row = registry.conn.execute(
        """
        SELECT review_method, review_notes, review_policy_id, review_policy_version
        FROM keypoint_quality
        WHERE dataset_id = ? AND refined_run = ?;
        """,
        ("dataset_keypoint", "refined_legacy"),
    ).fetchone()
    assert row is not None
    assert str(row["review_method"]) == "manual"
    assert str(row["review_notes"]) == "added via ensure_columns"
    assert str(row["review_policy_id"]) == "keypoint_auto_review_v1"
    assert int(row["review_policy_version"]) == 1
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
        "review_method",
        "review_intended_use",
        "review_policy_id",
        "review_policy_version",
        "usable_keypoints",
        "total_keypoints",
        "usable_keypoints_rate",
        "quality_updated_utc",
        "zarr_mtime_ns",
        "quality_stale",
    }
    assert expected_columns.issubset(set(row.keys()))
    registry.close()
