from __future__ import annotations

import json
from pathlib import Path

from fisheye.registry.db import Registry
from fisheye.utils.dish_mask_registry_sync import sync_dish_mask_registry_status


def _insert_dataset(registry: Registry, *, dataset_id: str, zarr_path: Path, recording_id: str) -> None:
    with registry.conn:
        registry.conn.execute(
            """
            INSERT INTO datasets (
                dataset_id,
                recording_id,
                zarr_path,
                status,
                created_utc,
                last_seen_utc
            )
            VALUES (?, ?, ?, 'ok', '2026-06-16T00:00:00+00:00', '2026-06-16T00:00:00+00:00');
            """,
            (dataset_id, recording_id, str(zarr_path)),
        )


def test_sync_dish_mask_registry_status_marks_dataset_ok(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "recording" / "zarr" / "example_analysis.zarr"
    zarr_path.mkdir(parents=True)
    registry = Registry(registry_path)
    _insert_dataset(registry, dataset_id="dataset_1", zarr_path=zarr_path, recording_id="recording_1")
    registry.close()

    result = sync_dish_mask_registry_status(
        zarr_path,
        registry_path,
        method="manual_rectangle",
        source="unit_test",
        details={"shape": "rectangle", "array_name": "images_ds", "frame_index": 123},
    )

    assert result.synced is True
    assert result.dataset_id == "dataset_1"
    registry = Registry(registry_path)
    try:
        row = registry.conn.execute(
            """
            SELECT dataset_id, recording_id, step_name, status, method, details_json, source
            FROM recording_step_status
            WHERE dataset_id = ? AND step_name = 'dish_mask';
            """,
            ("dataset_1",),
        ).fetchone()
        assert row is not None
        assert row["recording_id"] == "recording_1"
        assert row["status"] == "ok"
        assert row["method"] == "manual_rectangle"
        assert row["source"] == "unit_test"
        details = json.loads(row["details_json"])
        assert details["reason"] == "dish_mask_saved"
        assert details["shape"] == "rectangle"
        assert details["array_name"] == "images_ds"
        assert details["frame_index"] == 123

        history_count = registry.conn.execute(
            """
            SELECT COUNT(*) AS n
            FROM recording_step_status_history
            WHERE dataset_id = ? AND step_name = 'dish_mask';
            """,
            ("dataset_1",),
        ).fetchone()["n"]
        assert history_count == 1
    finally:
        registry.close()


def test_sync_dish_mask_registry_status_reports_missing_dataset(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    zarr_path = tmp_path / "unregistered.zarr"
    zarr_path.mkdir()
    Registry(registry_path).close()

    result = sync_dish_mask_registry_status(zarr_path, registry_path)

    assert result.synced is False
    assert result.status == "dataset_not_found"
    registry = Registry(registry_path)
    try:
        rows = registry.conn.execute("SELECT * FROM recording_step_status;").fetchall()
        assert rows == []
    finally:
        registry.close()
