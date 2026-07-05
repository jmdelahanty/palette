"""Tests for the reconcile orchestrator (Registry.reconcile_dataset_from_root).

Covers the checkpoint-1 validation bar:
- reconcile populates the detection/keypoint data-profile tables,
- reconcile after register is a no-op on register-owned tables (strict superset),
- reconcile is idempotent (run twice -> identical registry state), and
- include_step_status refreshes recording_step_status idempotently (no history
  growth on a converged dataset).
"""

from __future__ import annotations

from pathlib import Path
import sys

import zarr

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from fisheye.registry.db import Registry
from fisheye.registry import maintenance as maintenance_cli

_VOLATILE_SUFFIXES = ("updated_utc", "seen_utc")
_VOLATILE_COLUMNS = {"recorded_utc"}
_PROFILE_TABLES = {"detection_data_profile", "keypoint_data_profile"}


def _detection_summary() -> dict:
    return {
        "schema_name": "detection_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-12T03:00:00+00:00",
        "dataset": {"recording_id": "rec_a", "zarr_use": "training"},
        "source": {"detection_path": "refined_detect_runs/r/manual", "detection_type": "manual"},
        "coverage": {"frames_total": 100, "frames_with_detections": 95, "coverage_percent": 95.0},
        "counts": {"detections_total": 950, "detections_per_frame": {"p50": 9.0, "p90": 10.0}},
        "geometry_norm": {
            "w": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "h": {"p10": 0.1, "p50": 0.2, "p90": 0.3},
            "area": {"p10": 0.01, "p50": 0.04, "p90": 0.09},
            "aspect_ratio": {"p10": 0.8, "p50": 1.0, "p90": 1.2},
        },
        "spatial": {"edge_proximity_rate": 0.03},
        "composition": {"rig_id": "rig_a", "protocol_name": "DefaultScreen", "dpf_at_acquisition": 7},
    }


def _keypoint_summary() -> dict:
    return {
        "schema_name": "keypoint_dataset_profile",
        "schema_version": "v1",
        "created_at_utc": "2026-02-12T03:00:00+00:00",
        "dataset": {"recording_id": "rec_a", "zarr_use": "training"},
        "source": {"keypoint_method": "yolo", "keypoint_run": "run_a", "skeleton_id": "skel_v1"},
        "quality": {
            "rows_total": 100,
            "rows_usable": 90,
            "usable_keypoints_total": 630,
            "usable_rate": 0.9,
            "confidence_valid_rate": 0.95,
            "geometry_valid_rate": 0.92,
        },
        "geometry": {
            "triangle_area": {"stats": {"p10": 1.0, "p50": 2.0, "p90": 3.0}},
            "min_angle": {"stats": {"p10": 10.0, "p50": 20.0, "p90": 30.0}},
            "heading": {"stats": {"p10": 0.1, "p50": 0.2, "p90": 0.3}},
        },
        "composition": {"rig_id": "rig_a", "protocol_name": "DefaultScreen", "dpf_at_acquisition": 7},
    }


def _build_zarr(zarr_path: Path) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["session_uuid"] = "session-abc123"
    root.attrs["recording_id"] = "rec_a"
    root.attrs["zarr_use"] = "training"
    root.attrs["zarr_purpose"] = "training"

    analysis = root.require_group("analysis")
    det_parent = analysis.require_group("detection_profile_runs")
    det_run = det_parent.require_group("detection_profile_2026-02-12_03-00-00")
    det_run.attrs["profile_summary"] = _detection_summary()
    det_run.attrs["created_at_utc"] = "2026-02-12T03:00:00+00:00"
    det_parent.attrs["latest"] = "detection_profile_2026-02-12_03-00-00"

    kpt_parent = analysis.require_group("keypoint_profile_runs")
    kpt_run = kpt_parent.require_group("keypoint_profile_2026-02-12_03-00-00")
    kpt_run.attrs["profile_summary"] = _keypoint_summary()
    kpt_run.attrs["created_at_utc"] = "2026-02-12T03:00:00+00:00"
    kpt_parent.attrs["latest"] = "keypoint_profile_2026-02-12_03-00-00"


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


def test_reconcile_populates_profiles_and_is_register_no_op(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    registry_path = tmp_path / "registry.sqlite"

    registry = Registry(registry_path)
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        dataset_id = registry.register_from_root(root, zarr_path)

        # Register does not populate the data-profile tables.
        assert registry.conn.execute(
            "SELECT COUNT(*) AS n FROM detection_data_profile;"
        ).fetchone()["n"] == 0

        register_snapshot = _dump(registry, exclude_tables=_PROFILE_TABLES)

        result = registry.reconcile_dataset_from_root(root, zarr_path)
        assert result["dataset_id"] == dataset_id
        assert result["detection_data_profile_rows"] == 1
        assert result["keypoint_data_profile_rows"] == 1

        # Reconcile after register is a no-op on register-owned tables.
        assert _dump(registry, exclude_tables=_PROFILE_TABLES) == register_snapshot

        # Profile tables are now populated.
        det_rows = registry.conn.execute(
            "SELECT * FROM detection_data_profile WHERE dataset_id = ?;", (dataset_id,)
        ).fetchall()
        assert len(det_rows) == 1
        assert det_rows[0]["coverage_percent"] == 95.0
        kpt_rows = registry.conn.execute(
            "SELECT * FROM keypoint_data_profile WHERE dataset_id = ?;", (dataset_id,)
        ).fetchall()
        assert len(kpt_rows) == 1
        assert kpt_rows[0]["rows_total"] == 100
    finally:
        registry.close()


def test_reconcile_is_idempotent(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    registry_path = tmp_path / "registry.sqlite"

    registry = Registry(registry_path)
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        registry.reconcile_dataset_from_root(root, zarr_path)
        first = _dump(registry)
        registry.reconcile_dataset_from_root(root, zarr_path)
        second = _dump(registry)
        assert first == second
    finally:
        registry.close()


def test_reconcile_dataset_cli(tmp_path: Path, capsys) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    registry_path = tmp_path / "registry.sqlite"

    maintenance_cli.main(
        ["--registry", str(registry_path), "--reconcile-dataset", str(zarr_path)]
    )
    out = capsys.readouterr().out
    assert "Reconciled dataset" in out
    assert "detection_data_profile rows=1" in out

    registry = Registry(registry_path)
    try:
        n = registry.conn.execute(
            "SELECT COUNT(*) AS n FROM detection_data_profile;"
        ).fetchone()["n"]
        assert n == 1
        n_kpt = registry.conn.execute(
            "SELECT COUNT(*) AS n FROM keypoint_data_profile;"
        ).fetchone()["n"]
        assert n_kpt == 1
    finally:
        registry.close()


def test_reconcile_include_step_status_is_idempotent(tmp_path: Path) -> None:
    zarr_path = tmp_path / "dataset_training.zarr"
    _build_zarr(zarr_path)
    registry_path = tmp_path / "registry.sqlite"

    registry = Registry(registry_path)
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        first = registry.reconcile_dataset_from_root(root, zarr_path, include_step_status=True)
        assert "recording_step_status" in first
        step_count = registry.conn.execute(
            "SELECT COUNT(*) AS n FROM recording_step_status;"
        ).fetchone()["n"]
        assert step_count > 0

        second = registry.reconcile_dataset_from_root(root, zarr_path, include_step_status=True)
        # Converged dataset: no new history rows appended on the second pass.
        assert second["recording_step_status"]["history_rows_inserted"] == 0
    finally:
        registry.close()
