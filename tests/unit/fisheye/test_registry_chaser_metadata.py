from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.registry.chaser_metadata import (
    apply_chaser_metadata_census,
    build_chaser_metadata_census,
    select_census_datasets,
)
from fisheye.registry.db import Registry
from fisheye.registry.extractors.chaser_metadata import extract_recording_chaser_metadata


def _protocol(chasers: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "protocol_name": "RedScare",
            "steps": [{"parameters": {"chasers": chasers}}],
        }
    )


def _write_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "rec_a_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update(
        {
            "dataset_id": "dataset_a",
            "recording_id": "rec_a",
            "session_uuid": "rec_a",
            "zarr_purpose": "analysis",
            "zarr_use": "analysis",
            "recording_name": "rec_a",
            "recording_type": "behavior",
            "protocol_name": "RedScare",
        }
    )
    parent = root.require_group("analysis/stimulus_runs")
    one = parent.create_group("stimulus_one")
    one.attrs["protocol_json"] = _protocol(
        [{"enable_chase": True, "enable_random_movement": False}]
    )
    three = parent.create_group("stimulus_three")
    three.attrs["protocol_json"] = _protocol(
        [
            {"enable_chase": True, "enable_random_movement": False},
            {"enable_chase": False, "enable_random_movement": True},
            {"enable_chase": False, "enable_random_movement": False},
        ]
    )
    parent.attrs["latest"] = "stimulus_three"
    return zarr_path


def test_extract_recording_chasers_uses_rows_not_list_columns(tmp_path: Path) -> None:
    zarr_path = _write_archive(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)

    extraction = extract_recording_chaser_metadata(
        root,
        zarr_path=zarr_path,
        recording_id="rec_a",
    )

    assert extraction.stimulus_run_count == 2
    assert not extraction.issues
    assert len(extraction.rows) == 4
    assert [
        (row["stimulus_run_id"], row["chaser_index"], row["behavior_class"])
        for row in extraction.rows
    ] == [
        ("stimulus_one", 0, "aggressive"),
        ("stimulus_three", 0, "aggressive"),
        ("stimulus_three", 1, "random_non_chasing"),
        ("stimulus_three", 2, "inert"),
    ]
    assert all(not isinstance(row["behavior_class"], list) for row in extraction.rows)


def test_registry_scan_and_explicit_census_backfill_recording_chasers(tmp_path: Path) -> None:
    zarr_path = _write_archive(tmp_path)
    registry_path = tmp_path / "registry.sqlite"
    registry = Registry(registry_path)
    try:
        dataset_id = registry.scan_zarr(zarr_path)
        rows = registry.conn.execute(
            """
            SELECT stimulus_run_id, chaser_index, behavior_class
            FROM recording_chasers
            WHERE dataset_id = ?
            ORDER BY stimulus_run_id, chaser_index
            """,
            (dataset_id,),
        ).fetchall()
        registry.conn.execute("DELETE FROM recording_chasers WHERE dataset_id = ?", (dataset_id,))
        registry.conn.commit()
    finally:
        registry.close()

    assert len(rows) == 4
    datasets = select_census_datasets(registry_path, protocol_name="redscare")
    census = build_chaser_metadata_census(datasets)
    assert census["dataset_count"] == 1
    assert census["recording_count"] == 1
    assert census["physical_archive_count"] == 1
    assert census["behavior_counts"] == {
        "aggressive": 2,
        "inert": 1,
        "random_non_chasing": 1,
    }
    assert census["stimulus_runs_by_chaser_count"] == {"1": 1, "3": 1}
    assert census["issue_count"] == 0
    assert apply_chaser_metadata_census(registry_path, census) == 1

    registry = Registry(registry_path)
    try:
        restored = registry.conn.execute(
            "SELECT COUNT(*) AS count FROM recording_chasers WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    finally:
        registry.close()
    assert restored is not None
    assert int(restored["count"]) == 4
