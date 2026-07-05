"""Checkpoint-2 tests for the subject_mask data-profile path.

Covers the brief's validation bar:
- profile builder -> zarr (``analysis/subject_mask_profile_runs``) -> extractor
  -> ``subject_mask_data_profile`` table round trip on a synthetic fixture,
- reconcile idempotency with the new extractor (run twice -> identical state),
- reconcile-after-register stays a no-op on register-owned tables,
- ``*_latest`` / ``recording_*_latest`` view semantics,
- maintenance CLI (--reconcile-dataset) flows through the new extractor.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry import maintenance as maintenance_cli
from fisheye.utils.subject_mask_profile import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_subject_mask_profile_summary,
    write_subject_mask_profile,
)

_VOLATILE_SUFFIXES = ("updated_utc", "seen_utc")
_VOLATILE_COLUMNS = {"recorded_utc"}
_PROFILE_TABLES = {
    "detection_data_profile",
    "keypoint_data_profile",
    "subject_mask_data_profile",
}


def _build_zarr(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["session_uuid"] = "session-abc123"
    root.attrs["recording_id"] = "rec_a"
    root.attrs["zarr_use"] = "training"
    root.attrs["zarr_purpose"] = "training"

    labels = ["subject_body", "eyes_union", "swim_bladder"]
    parent = root.require_group("refined_subject_masks_runs")
    run = parent.require_group("masks_v1")
    run.attrs.update(
        {
            "created_at_utc": "2026-02-12T03:00:00+00:00",
            "method": "unet_v2",
            "label_schema_id": "subject_v1_union",
            "mask_labels": labels,
            "run_semantics": "full",
            "source_crop_run": "crop_a",
            "source_keypoints_run": "kp_a",
            "source_subject_mask_run": "raw_masks_v1",
            "total_rois": 4,
            "refined_subject_mask_review_status": {
                "state": "approved",
                "method": "manual",
                "intended_use": "training",
                "timestamp_utc": "2026-02-12T04:00:00+00:00",
            },
        }
    )
    run.create_array("available_channels", data=np.asarray([True, True, True], dtype=bool))
    # 4 ROIs x 3 components x 4x4 pixels:
    # subject_body fills half the ROI in every row; eyes_union is 1px in 3 rows;
    # swim_bladder is 1px in 2 rows.
    masks = np.zeros((4, 3, 4, 4), dtype=np.uint8)
    masks[:, 0, :, :2] = 1
    masks[:3, 1, 0, 0] = 1
    masks[:2, 2, 1, :1] = 1
    run.create_array("masks_roi", data=masks)
    parent.attrs["latest"] = "masks_v1"


def _write_profile(zarr_path: Path, *, run_name: str = "subject_mask_profile_2026-02-12_05-00-00") -> None:
    root = zarr.open_group(str(zarr_path), mode="r+")
    write_subject_mask_profile(
        root,
        zarr_path=zarr_path,
        run_name=run_name,
        created_at_utc="2026-02-12T05:00:00+00:00",
    )


def _table_names(registry: Registry) -> list[str]:
    rows = registry.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _is_volatile(column: str) -> bool:
    if column in _VOLATILE_COLUMNS:
        return True
    return any(column.endswith(suffix) for suffix in _VOLATILE_SUFFIXES)


def _dump(registry: Registry, *, exclude_tables: set[str] | None = None) -> dict:
    exclude = exclude_tables or set()
    snapshot: dict = {}
    for table in _table_names(registry):
        if table in exclude or table.startswith("sqlite_"):
            continue
        cursor = registry.conn.execute(f"SELECT * FROM {table};")
        columns = [desc[0] for desc in cursor.description]
        keep = [col for col in columns if not _is_volatile(col)]
        rows = []
        for row in cursor.fetchall():
            rows.append(tuple((col, row[col]) for col in keep))
        snapshot[table] = sorted(rows)
    return snapshot


def test_builder_summary_sections_and_stats(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)

    root = zarr.open_group(str(zarr_path), mode="r")
    summary = build_subject_mask_profile_summary(root, zarr_path=zarr_path)

    assert summary["schema_name"] == SCHEMA_NAME == "subject_mask_dataset_profile"
    assert summary["schema_version"] == SCHEMA_VERSION == "v1"
    for section in ("dataset", "source", "coverage", "components"):
        assert section in summary

    assert summary["dataset"]["recording_id"] == "rec_a"
    assert summary["dataset"]["zarr_use"] == "training"
    assert summary["source"]["mask_path"] == "refined_subject_masks_runs/masks_v1"
    assert summary["source"]["subject_mask_method"] == "unet_v2"
    assert summary["source"]["label_schema_id"] == "subject_v1_union"
    assert summary["source"]["eye_component_mode"] == "union"
    assert summary["source"]["review_state"] == "approved"

    coverage = summary["coverage"]
    assert coverage["total_rois"] == 4
    assert coverage["rows_with_any_mask"] == 4
    assert coverage["coverage_percent"] == 100.0
    assert coverage["available_component_count"] == 3
    assert coverage["rows_with_all_available_components"] == 2

    body = summary["components"]["subject_body"]
    assert body["presence_count"] == 4
    assert body["presence_rate"] == 1.0
    assert body["area_px"]["p50"] == 8.0
    assert body["area_norm"]["p50"] == 0.5

    eyes = summary["components"]["eyes_union"]
    assert eyes["presence_count"] == 3
    assert eyes["presence_rate"] == 0.75
    assert eyes["area_norm"]["p50"] == 1.0 / 16.0

    bladder = summary["components"]["swim_bladder"]
    assert bladder["presence_count"] == 2
    assert bladder["presence_rate"] == 0.5


def test_profile_round_trip_builder_zarr_extractor_table(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    _write_profile(zarr_path)

    # Zarr convention: run group + profile_summary attr + latest pointer.
    root = zarr.open_group(str(zarr_path), mode="r")
    runs_parent = root["analysis"]["subject_mask_profile_runs"]
    assert runs_parent.attrs["latest"] == "subject_mask_profile_2026-02-12_05-00-00"
    run_group = runs_parent["subject_mask_profile_2026-02-12_05-00-00"]
    assert run_group.attrs["schema_name"] == "subject_mask_dataset_profile"
    assert isinstance(dict(run_group.attrs)["profile_summary"], dict)

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = registry.reconcile_dataset_from_root(root, zarr_path)
        assert result["subject_mask_data_profile_rows"] == 1

        row = registry.conn.execute(
            "SELECT * FROM subject_mask_data_profile;"
        ).fetchone()
        assert row is not None
        assert row["dataset_id"] == result["dataset_id"]
        assert row["profile_run"] == "subject_mask_profile_2026-02-12_05-00-00"
        assert row["recording_id"] == "rec_a"
        assert row["zarr_use"] == "training"
        assert row["subject_mask_method"] == "unet_v2"
        assert row["label_schema_id"] == "subject_v1_union"
        assert row["source_keypoints_run"] == "kp_a"
        assert row["source_crop_run"] == "crop_a"
        assert row["run_semantics"] == "full"
        assert row["profile_created_utc"] == "2026-02-12T05:00:00+00:00"
        assert row["total_rois"] == 4
        assert row["rows_with_any_mask"] == 4
        assert row["coverage_percent"] == 100.0
        assert row["available_component_count"] == 3
        assert row["subject_body_presence_rate"] == 1.0
        assert row["subject_body_area_p10"] == 0.5
        assert row["subject_body_area_p50"] == 0.5
        assert row["subject_body_area_p90"] == 0.5
        assert row["eyes_union_presence_rate"] == 0.75
        assert row["eyes_union_area_p50"] == 1.0 / 16.0
        assert row["swim_bladder_presence_rate"] == 0.5
        # Labels absent from the run stay NULL (union schema has no eye_left/right).
        assert row["eye_left_presence_rate"] is None
        assert row["eye_right_area_p50"] is None
        # The archived summary round-trips through profile_json.
        archived = json.loads(row["profile_json"])
        assert archived["schema_name"] == "subject_mask_dataset_profile"
        assert archived["components"]["swim_bladder"]["presence_count"] == 2
    finally:
        registry.close()


def test_reconcile_is_idempotent_with_subject_mask_profile(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    _write_profile(zarr_path)

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        registry.reconcile_dataset_from_root(root, zarr_path)
        first = _dump(registry)
        registry.reconcile_dataset_from_root(root, zarr_path)
        second = _dump(registry)
        assert first == second
        assert len(first["subject_mask_data_profile"]) == 1
    finally:
        registry.close()


def test_reconcile_after_register_is_no_op_on_register_tables(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    _write_profile(zarr_path)

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        registry.register_from_root(root, zarr_path)

        # Register does not populate the profile table; reconcile does.
        assert registry.conn.execute(
            "SELECT COUNT(*) AS n FROM subject_mask_data_profile;"
        ).fetchone()["n"] == 0
        register_snapshot = _dump(registry, exclude_tables=_PROFILE_TABLES)

        registry.reconcile_dataset_from_root(root, zarr_path)
        assert _dump(registry, exclude_tables=_PROFILE_TABLES) == register_snapshot
        assert registry.conn.execute(
            "SELECT COUNT(*) AS n FROM subject_mask_data_profile;"
        ).fetchone()["n"] == 1
    finally:
        registry.close()


def test_latest_views_pick_newest_profile_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    _write_profile(zarr_path, run_name="subject_mask_profile_2026-02-12_05-00-00")

    root = zarr.open_group(str(zarr_path), mode="r+")
    write_subject_mask_profile(
        root,
        zarr_path=zarr_path,
        run_name="subject_mask_profile_2026-02-13_05-00-00",
        created_at_utc="2026-02-13T05:00:00+00:00",
    )

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        result = registry.reconcile_dataset_from_root(
            zarr.open_group(str(zarr_path), mode="r"), zarr_path
        )
        assert result["subject_mask_data_profile_rows"] == 2

        latest = registry.query_subject_mask_data_profile_latest()
        assert len(latest) == 1
        assert latest[0]["profile_run"] == "subject_mask_profile_2026-02-13_05-00-00"

        recording_latest = registry.query_recording_subject_mask_data_profile_latest(
            recording_ids=["rec_a"]
        )
        assert len(recording_latest) == 1
        assert recording_latest[0]["profile_run"] == "subject_mask_profile_2026-02-13_05-00-00"
        assert recording_latest[0]["zarr_path"] == str(zarr_path)
    finally:
        registry.close()


def test_reconcile_dataset_cli_reports_subject_mask_profile(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    _write_profile(zarr_path)
    registry_path = tmp_path / "registry.sqlite"

    maintenance_cli.main(
        ["--registry", str(registry_path), "--reconcile-dataset", str(zarr_path)]
    )
    out = capsys.readouterr().out
    assert "subject_mask_data_profile rows=1" in out

    registry = Registry(registry_path)
    try:
        n = registry.conn.execute(
            "SELECT COUNT(*) AS n FROM subject_mask_data_profile;"
        ).fetchone()["n"]
        assert n == 1
    finally:
        registry.close()
