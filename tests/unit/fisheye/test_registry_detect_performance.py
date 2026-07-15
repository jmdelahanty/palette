"""Unit tests for detect performance registry extraction and views."""

import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry, _extract_detect_performance_rows, _extract_detect_quality_rows


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


def _build_fake_detect_quality_root(*, timestamp_key: str = "timestamp_utc") -> _FakeGroup:
    root = _FakeGroup()
    detect_parent = root.add_group("detect_runs")
    detect_parent.add_group("detect_001", attrs={"detection_method": "yolo"})

    refined_parent = root.add_group("refined_detect_runs")
    refined = refined_parent.add_group(
        "refined_001",
        attrs={
            "source_detect_run": "detect_001",
            "created_utc": "2026-02-10T00:00:00+00:00",
            "detect_review_status": {
                "state": "approved",
                "method": "manual",
                "intended_use": "training",
                "reviewer": "reviewer_detect",
                "notes": "checked",
                "resolved_group": "refined",
                timestamp_key: "2026-02-10T00:01:00+00:00",
            },
        },
    )
    instances = refined.add_group("instances")
    instances.add_array("bbox_norm_coords", np.zeros((4, 4), dtype=np.float32))
    instances.add_array("source_kind_codes", np.array([1, 1, 2, 1], dtype=np.int8))
    return root


def _build_fake_detect_performance_root() -> _FakeGroup:
    root = _FakeGroup()
    detect_parent = root.add_group("detect_runs")
    detect_parent.add_group(
        "detect_001",
        attrs={
            "created_at_utc": "2026-02-10T00:10:00+00:00",
            "detect_timestamp_utc": "2026-02-10T00:09:00+00:00",
            "created_utc": "2026-02-10T00:08:00+00:00",
            "timestamp_utc": "2026-02-10T00:07:00+00:00",
            "detection_method": "yolo",
            "provenance": {"created_at_utc": "2026-02-10T00:06:00+00:00"},
        },
    ).add_array("frame_counts", np.array([1, 0, 1, 1], dtype=np.int32))
    return root


def _detect_quality_record(
    *,
    refined_run: str,
    source_detect_run: str = "detect_001",
    detect_method: str = "yolo",
    refined_created_utc: str | None = "2026-02-10T00:00:00+00:00",
    review_timestamp_utc: str | None = "2026-02-10T00:01:00+00:00",
    quality_updated_utc: str = "2026-02-10T00:02:00+00:00",
) -> dict[str, object | None]:
    return {
        "refined_run": refined_run,
        "refined_created_utc": refined_created_utc,
        "source_detect_run": source_detect_run,
        "detect_method": detect_method,
        "review_state": "approved",
        "review_method": "manual",
        "review_intended_use": "training",
        "review_reviewer": "reviewer_detect",
        "review_notes": "checked",
        "review_timestamp_utc": review_timestamp_utc,
        "review_resolved_group": "filtered",
        "total_detections": 10,
        "real_detections": 8,
        "interpolated_detections": 2,
        "interpolated_detections_rate": 0.2,
        "quality_updated_utc": quality_updated_utc,
        "zarr_mtime_ns": 123456789,
    }


def test_extract_detect_quality_rows_populates_shared_review_fields(tmp_path: Path) -> None:
    rows = _extract_detect_quality_rows(_build_fake_detect_quality_root(), zarr_path=tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["review_state"] == "approved"
    assert row["review_method"] == "manual"
    assert row["review_intended_use"] == "training"
    assert row["review_reviewer"] == "reviewer_detect"
    assert row["review_notes"] == "checked"
    assert row["review_timestamp_utc"] == "2026-02-10T00:01:00+00:00"
    assert row["review_resolved_group"] == "refined"
    assert row["interpolated_detections_rate"] == 0.25


def test_extract_detect_performance_rows_prefers_created_at_utc(tmp_path: Path) -> None:
    rows = _extract_detect_performance_rows(
        _build_fake_detect_performance_root(),
        zarr_path=tmp_path,
        recording_id="recording_detect",
        zarr_use="analysis",
    )
    assert len(rows) == 1
    assert rows[0]["detect_created_utc"] == "2026-02-10T00:10:00+00:00"


def test_extract_detect_quality_rows_prefers_created_at_utc(tmp_path: Path) -> None:
    root = _build_fake_detect_quality_root()
    refined = root["refined_detect_runs"]["refined_001"]
    refined.attrs["created_at_utc"] = "2026-02-10T00:10:00+00:00"
    refined.attrs["refinement_timestamp"] = "2026-02-10T00:09:00+00:00"
    rows = _extract_detect_quality_rows(root, zarr_path=tmp_path)
    assert len(rows) == 1
    assert rows[0]["refined_created_utc"] == "2026-02-10T00:10:00+00:00"


def test_detect_quality_legacy_timestamp_alias_maps_to_review_timestamp_utc(tmp_path: Path) -> None:
    rows = _extract_detect_quality_rows(
        _build_fake_detect_quality_root(timestamp_key="reviewed_at_utc"),
        zarr_path=tmp_path,
    )
    assert len(rows) == 1
    assert rows[0]["review_timestamp_utc"] == "2026-02-10T00:01:00+00:00"


def test_extract_detect_quality_rows_multiple_runs_with_mixed_review_payloads(tmp_path: Path) -> None:
    root = _FakeGroup()
    detect_parent = root.add_group("detect_runs")
    detect_parent.add_group("detect_001", attrs={"detection_method": "yolo"})
    detect_parent.add_group("detect_002", attrs={"method": "traditional"})
    detect_parent.add_group("detect_003")

    refined_parent = root.add_group("refined_detect_runs")
    refined_full = refined_parent.add_group(
        "refined_full",
        attrs={
            "source_detect_run": "detect_001",
            "created_utc": "2026-02-10T00:00:00+00:00",
            "detect_review_status": {
                "state": "approved",
                "method": "manual",
                "intended_use": "training",
                "reviewer": "alice",
                "notes": "full payload",
                "timestamp_utc": "2026-02-10T00:05:00+00:00",
                "resolved_group": "filtered",
            },
        },
    )
    full_filtered = refined_full.add_group("filtered")
    full_filtered.add_array("bbox_norm_coords", np.zeros((3, 4), dtype=np.float32))
    full_filtered.add_array("detection_source", np.array([0, 1, 0], dtype=np.int8))

    refined_json = refined_parent.add_group(
        "refined_json",
        attrs={
            "source_detect_run": "detect_002",
            "refinement_timestamp": "2026-02-10T01:00:00+00:00",
            "created_utc": "2026-02-10T00:59:00+00:00",
            "manual_review_latest": "manual",
            "detect_review_status": json.dumps(
                {
                    "state": "needs_review",
                    "intended_use": "validation",
                    "reviewer": "bob",
                    "notes": "json payload",
                    "reviewed_at": "2026-02-10T01:05:00+00:00",
                }
            ),
        },
    )
    refined_json.add_group(
        "manual",
        attrs={
            "total_detections": 5,
            "interpolated_detections": 2,
            "original_detections": 3,
        },
    )

    refined_fallback = refined_parent.add_group(
        "refined_fallback",
        attrs={
            "source_detect_run": "detect_003",
            "timestamp_utc": "2026-02-10T02:00:00+00:00",
            "detect_review_status": "not-json",
        },
    )
    refined_fallback.add_group(
        "interpolated",
        attrs={
            "total_detections": 4,
            "interpolated_detections": 1,
        },
    )
    refined_parent.add_group("refined_skip", attrs={"created_utc": "2026-02-10T03:00:00+00:00"})

    rows = _extract_detect_quality_rows(root, zarr_path=tmp_path)
    by_run = {str(row["refined_run"]): row for row in rows}
    assert set(by_run.keys()) == {"refined_fallback", "refined_full", "refined_json"}

    full_row = by_run["refined_full"]
    assert full_row["detect_method"] == "yolo"
    assert full_row["review_state"] == "approved"
    assert full_row["review_method"] == "manual"
    assert full_row["review_intended_use"] == "training"
    assert full_row["review_reviewer"] == "alice"
    assert full_row["review_notes"] == "full payload"
    assert full_row["review_timestamp_utc"] == "2026-02-10T00:05:00+00:00"
    assert full_row["review_resolved_group"] == "filtered"
    assert full_row["total_detections"] == 3
    assert full_row["real_detections"] == 2
    assert full_row["interpolated_detections"] == 1
    assert full_row["interpolated_detections_rate"] == 1.0 / 3.0

    json_row = by_run["refined_json"]
    assert json_row["refined_created_utc"] == "2026-02-10T01:00:00+00:00"
    assert json_row["detect_method"] == "traditional"
    assert json_row["review_state"] == "needs_review"
    assert json_row["review_method"] is None
    assert json_row["review_intended_use"] == "validation"
    assert json_row["review_reviewer"] == "bob"
    assert json_row["review_notes"] == "json payload"
    assert json_row["review_timestamp_utc"] == "2026-02-10T01:05:00+00:00"
    assert json_row["review_resolved_group"] == "manual"
    assert json_row["total_detections"] == 5
    assert json_row["real_detections"] == 3
    assert json_row["interpolated_detections"] == 2
    assert json_row["interpolated_detections_rate"] == 0.4

    fallback_row = by_run["refined_fallback"]
    assert fallback_row["refined_created_utc"] == "2026-02-10T02:00:00+00:00"
    assert fallback_row["detect_method"] is None
    assert fallback_row["review_state"] is None
    assert fallback_row["review_method"] is None
    assert fallback_row["review_intended_use"] is None
    assert fallback_row["review_reviewer"] is None
    assert fallback_row["review_notes"] is None
    assert fallback_row["review_timestamp_utc"] is None
    assert fallback_row["review_resolved_group"] == "interpolated"
    assert fallback_row["total_detections"] == 4
    assert fallback_row["real_detections"] == 3
    assert fallback_row["interpolated_detections"] == 1
    assert fallback_row["interpolated_detections_rate"] == 0.25

    assert full_row["quality_updated_utc"] == json_row["quality_updated_utc"] == fallback_row["quality_updated_utc"]
    assert isinstance(full_row["zarr_mtime_ns"], int)
    assert full_row["zarr_mtime_ns"] == json_row["zarr_mtime_ns"] == fallback_row["zarr_mtime_ns"]


def test_upsert_detect_quality_writes_shared_review_fields_to_current_view(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_detect", session_uuid="dataset_detect", zarr_path=tmp_path / "detect.zarr")
    registry.upsert_detect_quality(
        dataset_id="dataset_detect",
        refined_run="refined_001",
        refined_created_utc="2026-02-10T00:00:00+00:00",
        source_detect_run="detect_001",
        detect_method="yolo",
        review_state="approved",
        review_intended_use="training",
        review_reviewer="reviewer_detect",
        review_timestamp_utc="2026-02-10T00:01:00+00:00",
        review_resolved_group="filtered",
        total_detections=4,
        real_detections=3,
        interpolated_detections=1,
        interpolated_detections_rate=0.25,
        review_method="hybrid",
        review_notes="validated",
    )
    row = registry.query_detect_quality_current(
        dataset_ids=["dataset_detect"],
        detect_method="yolo",
    )[0]
    assert str(row["review_state"]) == "approved"
    assert str(row["review_method"]) == "hybrid"
    assert str(row["review_intended_use"]) == "training"
    assert str(row["review_reviewer"]) == "reviewer_detect"
    assert str(row["review_notes"]) == "validated"
    assert str(row["review_timestamp_utc"]) == "2026-02-10T00:01:00+00:00"
    assert str(row["review_resolved_group"]) == "filtered"
    registry.close()


def test_detect_quality_current_prefers_newest_review_timestamp_utc(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_detect", session_uuid="dataset_detect", zarr_path=tmp_path / "detect.zarr")
    registry.replace_detect_quality(
        "dataset_detect",
        [
            _detect_quality_record(
                refined_run="refined_older_review",
                refined_created_utc="2026-02-10T00:30:00+00:00",
                review_timestamp_utc="2026-02-10T00:35:00+00:00",
                quality_updated_utc="2026-02-10T00:40:00+00:00",
            ),
            _detect_quality_record(
                refined_run="refined_newer_review",
                refined_created_utc="2026-02-10T00:10:00+00:00",
                review_timestamp_utc="2026-02-10T00:45:00+00:00",
                quality_updated_utc="2026-02-10T00:20:00+00:00",
            ),
        ],
    )
    row = registry.query_detect_quality_current(
        dataset_ids=["dataset_detect"],
        detect_method="yolo",
    )[0]
    assert str(row["refined_run"]) == "refined_newer_review"
    assert str(row["review_timestamp_utc"]) == "2026-02-10T00:45:00+00:00"
    registry.close()


def test_detect_quality_current_falls_back_to_refined_created_when_review_timestamp_missing(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_detect", session_uuid="dataset_detect", zarr_path=tmp_path / "detect.zarr")
    registry.replace_detect_quality(
        "dataset_detect",
        [
            _detect_quality_record(
                refined_run="refined_z_older_created",
                refined_created_utc="2026-02-10T00:10:00+00:00",
                review_timestamp_utc=None,
                quality_updated_utc="2026-02-10T00:50:00+00:00",
            ),
            _detect_quality_record(
                refined_run="refined_a_newer_created",
                refined_created_utc="2026-02-10T00:20:00+00:00",
                review_timestamp_utc=None,
                quality_updated_utc="2026-02-10T00:50:00+00:00",
            ),
        ],
    )
    row = registry.query_detect_quality_current(
        dataset_ids=["dataset_detect"],
        detect_method="yolo",
    )[0]
    assert str(row["refined_run"]) == "refined_a_newer_created"
    assert str(row["refined_created_utc"]) == "2026-02-10T00:20:00+00:00"
    assert row["review_timestamp_utc"] is None
    registry.close()


def test_detect_quality_current_falls_back_to_lexical_refined_run_when_timestamps_tie(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_detect", session_uuid="dataset_detect", zarr_path=tmp_path / "detect.zarr")
    registry.replace_detect_quality(
        "dataset_detect",
        [
            _detect_quality_record(
                refined_run="refined_alpha",
                refined_created_utc=None,
                review_timestamp_utc=None,
                quality_updated_utc="2026-02-10T00:50:00+00:00",
            ),
            _detect_quality_record(
                refined_run="refined_zeta",
                refined_created_utc=None,
                review_timestamp_utc=None,
                quality_updated_utc="2026-02-10T00:50:00+00:00",
            ),
        ],
    )
    row = registry.query_detect_quality_current(
        dataset_ids=["dataset_detect"],
        detect_method="yolo",
    )[0]
    assert str(row["refined_run"]) == "refined_zeta"
    assert row["review_timestamp_utc"] is None
    assert row["refined_created_utc"] is None
    registry.close()


def test_detect_quality_current_prefers_reviewed_row_over_newer_unreviewed_row(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_detect", session_uuid="dataset_detect", zarr_path=tmp_path / "detect.zarr")
    reviewed = _detect_quality_record(
        refined_run="refined_reviewed",
        refined_created_utc="2026-02-10T00:00:00+00:00",
        review_timestamp_utc="2026-02-10T00:01:00+00:00",
        quality_updated_utc="2026-02-10T01:00:00+00:00",
    )
    unreviewed = _detect_quality_record(
        refined_run="refined_newer_unreviewed",
        refined_created_utc="2026-02-10T00:30:00+00:00",
        review_timestamp_utc=None,
        quality_updated_utc="2026-02-10T01:00:00+00:00",
    )
    unreviewed["review_state"] = None
    unreviewed["review_intended_use"] = None
    unreviewed["review_method"] = None
    unreviewed["review_timestamp_utc"] = None
    registry.replace_detect_quality(
        "dataset_detect",
        [reviewed, unreviewed],
    )
    row = registry.query_detect_quality_current(
        dataset_ids=["dataset_detect"],
        detect_method="yolo",
    )[0]
    assert str(row["refined_run"]) == "refined_reviewed"
    assert row["review_state"] == "approved"
    assert row["review_intended_use"] == "training"
    registry.close()


def test_refined_detect_review_current_alias_matches_detect_quality_current(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_detect", session_uuid="dataset_detect", zarr_path=tmp_path / "detect.zarr")
    registry.upsert_detect_quality(
        dataset_id="dataset_detect",
        refined_run="refined_001",
        refined_created_utc="2026-02-10T00:00:00+00:00",
        source_detect_run="detect_001",
        detect_method="yolo",
        review_state="approved",
        review_intended_use="training",
        review_reviewer="reviewer_detect",
        review_timestamp_utc="2026-02-10T00:01:00+00:00",
        review_resolved_group="refined",
        total_detections=4,
        real_detections=4,
        interpolated_detections=0,
        interpolated_detections_rate=0.0,
        review_method="manual",
        review_notes="approved curated refined surface",
    )
    old_rows = registry.query_detect_quality_current(dataset_ids=["dataset_detect"], detect_method="yolo")
    new_rows = registry.query_refined_detect_review_current(dataset_ids=["dataset_detect"], detect_method="yolo")
    assert len(old_rows) == 1
    assert len(new_rows) == 1
    old_row = dict(old_rows[0])
    new_row = dict(new_rows[0])
    assert new_row == old_row
    alias_count = registry.conn.execute(
        "SELECT COUNT(*) AS n FROM refined_detect_review_current WHERE dataset_id=?",
        ("dataset_detect",),
    ).fetchone()["n"]
    assert int(alias_count) == 1
    registry.close()


def test_refined_detect_review_current_query_falls_back_when_alias_view_missing(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_detect", session_uuid="dataset_detect", zarr_path=tmp_path / "detect.zarr")
    registry.upsert_detect_quality(
        dataset_id="dataset_detect",
        refined_run="refined_001",
        refined_created_utc="2026-02-10T00:00:00+00:00",
        source_detect_run="detect_001",
        detect_method="yolo",
        review_state="approved",
        review_intended_use="training",
        review_reviewer="reviewer_detect",
        review_timestamp_utc="2026-02-10T00:01:00+00:00",
        review_resolved_group="refined",
        total_detections=4,
        real_detections=4,
        interpolated_detections=0,
        interpolated_detections_rate=0.0,
        review_method="manual",
        review_notes="approved curated refined surface",
    )
    registry.conn.execute("DROP VIEW refined_detect_review_current;")
    registry.conn.commit()

    rows = registry.query_refined_detect_review_current(dataset_ids=["dataset_detect"], detect_method="yolo")

    assert len(rows) == 1
    assert rows[0]["review_state"] == "approved"
    assert rows[0]["review_intended_use"] == "training"
    registry.close()


def test_replace_detect_quality_auto_adds_review_columns_for_legacy_table(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    registry.upsert_dataset("dataset_detect", session_uuid="dataset_detect", zarr_path=tmp_path / "detect.zarr")
    registry.conn.execute("DROP VIEW IF EXISTS detect_quality_current;")
    registry.conn.execute("DROP TABLE detect_quality;")
    registry.conn.execute(
        """
        CREATE TABLE detect_quality (
            dataset_id TEXT NOT NULL,
            refined_run TEXT NOT NULL,
            refined_created_utc TEXT,
            source_detect_run TEXT NOT NULL,
            detect_method TEXT,
            review_state TEXT,
            review_intended_use TEXT,
            review_reviewer TEXT,
            review_timestamp_utc TEXT,
            review_resolved_group TEXT,
            total_detections INTEGER,
            real_detections INTEGER,
            interpolated_detections INTEGER,
            interpolated_detections_rate REAL,
            quality_updated_utc TEXT,
            zarr_mtime_ns INTEGER,
            PRIMARY KEY (dataset_id, refined_run),
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
        );
        """
    )
    registry.conn.commit()

    registry.replace_detect_quality(
        "dataset_detect",
        [
            {
                "refined_run": "refined_legacy",
                "refined_created_utc": "2026-02-10T00:00:00+00:00",
                "source_detect_run": "detect_001",
                "detect_method": "yolo",
                "review_state": "approved",
                "review_method": "manual",
                "review_intended_use": "training",
                "review_reviewer": "reviewer_detect",
                "review_notes": "added via ensure_columns",
                "review_timestamp_utc": "2026-02-10T00:01:00+00:00",
                "review_resolved_group": "filtered",
                "total_detections": 4,
                "real_detections": 3,
                "interpolated_detections": 1,
                "interpolated_detections_rate": 0.25,
            }
        ],
    )

    cols = {
        str(row["name"])
        for row in registry.conn.execute("PRAGMA table_info(detect_quality);").fetchall()
    }
    assert "review_method" in cols
    assert "review_notes" in cols
    row = registry.conn.execute(
        """
        SELECT review_method, review_notes
        FROM detect_quality
        WHERE dataset_id = ? AND refined_run = ?;
        """,
        ("dataset_detect", "refined_legacy"),
    ).fetchone()
    assert row is not None
    assert str(row["review_method"]) == "manual"
    assert str(row["review_notes"]) == "added via ensure_columns"
    registry.close()


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


def test_detect_performance_ignores_empty_legacy_run_group(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite")
    zarr_path = tmp_path / "recordings" / "rec_empty" / "zarr" / "rec_empty_analysis.zarr"
    _create_detect_archive(zarr_path, session_uuid="rec_empty_uuid")
    root = zarr.open_group(str(zarr_path), mode="a")
    root["detect_runs"].create_group("detect_empty_placeholder")

    dataset_id = registry.register_from_root(
        zarr.open_group(str(zarr_path), mode="r"),
        zarr_path,
    )

    rows = registry.conn.execute(
        """
        SELECT detect_run
        FROM detect_performance
        WHERE dataset_id = ?
        ORDER BY detect_run;
        """,
        (dataset_id,),
    ).fetchall()
    assert [str(row["detect_run"]) for row in rows] == ["detect_new", "detect_old"]

    latest = registry.conn.execute(
        """
        SELECT detect_run, coverage_percent
        FROM detect_performance_latest
        WHERE dataset_id = ?;
        """,
        (dataset_id,),
    ).fetchone()
    assert latest is not None
    assert str(latest["detect_run"]) == "detect_new"
    assert float(latest["coverage_percent"]) == 75.0
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
