"""Registry migration 076: realtime products and the models that ran."""

from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.registry.db import Registry
from fisheye.registry.extractors.realtime_products import extract_realtime_product_rows
from fisheye.registry.migrations import MIGRATION_METHODS
from fisheye.shared.acquisition_realtime_products import (
    build_realtime_products_record,
    write_realtime_products_record,
)

from tests.unit.fisheye.test_acquisition_realtime_products import _recording


def _root_with_record(tmp_path: Path, **kwargs) -> zarr.Group:
    rec, manifest = _recording(tmp_path, **kwargs)
    record = build_realtime_products_record(rec, manifest)
    root = zarr.open_group(str(tmp_path / "analysis.zarr"), mode="w")
    write_realtime_products_record(root, record)
    return root


def test_migration_076_is_registered_last() -> None:
    assert MIGRATION_METHODS[-1] == (
        76, "recording_realtime_products", "_migration_076_recording_realtime_products"
    )


def test_rows_mirror_the_record_and_the_view_joins_training_runs(tmp_path: Path) -> None:
    root = _root_with_record(tmp_path)
    rows = extract_realtime_product_rows(root, recording_id="rec-1")
    assert [row["product"] for row in rows] == ["detections", "pose"]
    detections = rows[0]
    assert (detections["row_count"], detections["result_rows"], detections["engine_precision"]) == (3, 3, "int8")

    registry = Registry(tmp_path / "registry.sqlite")
    try:
        assert registry._current_schema_version() == 76
        registry.conn.execute(
            "INSERT INTO training_runs (run_id, set_id, status, model_sha256) VALUES (?, ?, 'success', ?);",
            (detections["engine_manifest_run_id"], "set", detections["weights_sha256"]),
        )
        registry.replace_recording_realtime_products("dataset-1", rows)
        registry.replace_recording_realtime_products("dataset-1", rows)  # replaced whole, not duplicated
        registry.conn.commit()
        stored = registry.conn.execute(
            "SELECT COUNT(*) FROM recording_realtime_products WHERE dataset_id = 'dataset-1';"
        ).fetchone()[0]
        view = {
            row["product"]: dict(row)
            for row in registry.conn.execute("SELECT * FROM recording_realtime_models;")
        }
    finally:
        registry.close()
    assert stored == 2
    assert view["detections"]["training_run_by_weights"] == detections["engine_manifest_run_id"]
    assert view["detections"]["training_joins_agree"] == 1
    assert view["pose"]["training_run_by_weights"] is None  # no pose training run in this registry


def test_an_undeclared_session_is_one_row(tmp_path: Path) -> None:
    root = _root_with_record(tmp_path, declare=False)
    rows = extract_realtime_product_rows(root, recording_id="rec-1")
    assert [(row["product"], row["declared"], row["status"]) for row in rows] == [
        ("undeclared", 0, "undeclared")
    ]


def test_a_zarr_without_a_record_has_no_rows(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "empty.zarr"), mode="w")
    assert extract_realtime_product_rows(root, recording_id="rec-1") == []
