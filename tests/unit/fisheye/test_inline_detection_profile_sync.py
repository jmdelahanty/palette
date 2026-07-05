"""Tests for the relocated inline detection-profile sync helper.

``sync_latest_detection_profile_for_zarr`` moved from the retired
``utils/sync_detection_profile_registry.py`` into ``registry/inline_refresh.py``
and is now built on the shared detection profile extractor. It is the
approval/finalization-time single-run projection into ``detection_data_profile``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry.inline_refresh import sync_latest_detection_profile_for_zarr


def _summary() -> dict:
    return {
        "schema_name": "detection_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-12T03:00:00+00:00",
        "dataset": {"recording_id": "rec_summary", "zarr_use": "training"},
        "source": {"detection_path": "refined_detect_runs/r/manual", "detection_type": "manual"},
        "coverage": {"frames_total": 100, "frames_with_detections": 95, "coverage_percent": 95.0},
        "counts": {"detections_total": 950, "detections_per_frame": {"p50": 9.0, "p90": 10.0}},
        "geometry_norm": {"w": {"p10": 0.1, "p50": 0.2, "p90": 0.3}},
        "spatial": {"edge_proximity_rate": 0.03},
        "composition": {"rig_id": "rig_a", "protocol_name": "DefaultScreen", "dpf_at_acquisition": 7},
    }


def _build_zarr(zarr_path: Path, run_name: str) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    analysis = root.require_group("analysis")
    parent = analysis.require_group("detection_profile_runs")
    run_group = parent.require_group(run_name)
    run_group.attrs["profile_summary"] = _summary()
    run_group.attrs["created_at_utc"] = "2026-02-12T03:00:00+00:00"
    parent.attrs["latest"] = run_name


def test_inline_sync_upserts_latest_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_a_training.zarr"
    run_name = "detection_profile_2026-02-12_03-00-00"
    _build_zarr(zarr_path, run_name)
    registry_path = tmp_path / "registry.sqlite"

    registry = Registry(registry_path)
    try:
        registry.upsert_dataset(
            "dataset_a",
            session_uuid="dataset_a_session",
            zarr_path=zarr_path,
            recording_id="dataset_a_recording",
            artifact_kind="source_recording",
            zarr_use="training",
        )
        result = sync_latest_detection_profile_for_zarr(
            registry, zarr_path, dataset_id="dataset_a", apply=True
        )
        assert result["status"] == "updated"
        assert result["profile_run"] == run_name

        rows = registry.conn.execute(
            "SELECT * FROM detection_data_profile WHERE dataset_id = ?;", ("dataset_a",)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["profile_run"] == run_name
        assert rows[0]["coverage_percent"] == 95.0
        assert rows[0]["detection_type"] == "manual"
    finally:
        registry.close()


def test_inline_sync_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_b_training.zarr"
    _build_zarr(zarr_path, "detection_profile_2026-02-12_03-00-00")
    registry_path = tmp_path / "registry.sqlite"

    registry = Registry(registry_path)
    try:
        registry.upsert_dataset(
            "dataset_b",
            session_uuid="dataset_b_session",
            zarr_path=zarr_path,
            recording_id="dataset_b_recording",
            artifact_kind="source_recording",
            zarr_use="training",
        )
        result = sync_latest_detection_profile_for_zarr(
            registry, zarr_path, dataset_id="dataset_b", apply=False
        )
        assert result["status"] == "would_upsert"
        count = registry.conn.execute("SELECT COUNT(*) AS n FROM detection_data_profile;").fetchone()
        assert int(count["n"]) == 0
    finally:
        registry.close()


def test_inline_sync_missing_dataset_is_reported(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_c_training.zarr"
    _build_zarr(zarr_path, "detection_profile_2026-02-12_03-00-00")
    registry_path = tmp_path / "registry.sqlite"

    registry = Registry(registry_path)
    try:
        result = sync_latest_detection_profile_for_zarr(
            registry, zarr_path, dataset_id="unregistered", apply=True
        )
        assert result["status"] == "missing_dataset"
    finally:
        registry.close()
