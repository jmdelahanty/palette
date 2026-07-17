from __future__ import annotations

import json
from pathlib import Path

from fisheye.diagnostics.audit_analysis_storage_candidates import (
    build_storage_audit_report,
    discover_latest_completed_runs,
    render_markdown,
)


def _write_group(path: Path, attrs: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": attrs or {},
            }
        ),
        encoding="utf-8",
    )


def _write_array(
    path: Path,
    *,
    shape: list[int],
    chunks: list[int],
    shards: list[int] | None = None,
    payload_sizes: list[int] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    codecs = []
    if shards is not None:
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": chunks,
                    "codecs": [],
                    "index_codecs": [],
                    "index_location": "end",
                },
            }
        ]
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": shape,
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": shards or chunks},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0.0,
                "codecs": codecs,
                "attributes": {},
            }
        ),
        encoding="utf-8",
    )
    for index, size in enumerate(payload_sizes or []):
        payload = path / "c" / str(index)
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"x" * size)


def _make_archive(path: Path) -> Path:
    _write_group(path)
    eye_family = path / "analysis" / "eye_angle_runs"
    _write_group(eye_family, {"latest_complete": "eye_current"})
    _write_group(
        eye_family / "eye_current",
        {
            "schema_id": "palette.eye_angle.compact_dense.v2",
            "palette_run_completion_status": "complete",
            "cluster_output_staging": {"publish": "atomic"},
        },
    )
    _write_array(
        eye_family / "eye_current" / "frame_angles",
        shape=[20000, 64],
        chunks=[100, 64],
        payload_sizes=[100] * 40,
    )

    tail_family = path / "analysis" / "tail_kinematics_runs"
    _write_group(tail_family, {"latest_complete": "tail_current"})
    _write_group(tail_family / "tail_current")
    _write_array(
        tail_family / "tail_current" / "tail_angles",
        shape=[20000, 64],
        chunks=[100, 16],
        shards=[1000, 32],
        payload_sizes=[200, 220, 240],
    )

    shard_family = path / "keypoint_shard_runs"
    _write_group(shard_family, {"latest_complete": "shard_current"})
    _write_group(shard_family / "shard_current")
    _write_array(
        shard_family / "shard_current" / "points",
        shape=[1000, 2],
        chunks=[10, 2],
        payload_sizes=[10],
    )
    return path


def test_discover_latest_completed_runs_uses_authoritative_pointer(tmp_path: Path) -> None:
    zarr_path = _make_archive(tmp_path / "recording_analysis.zarr")

    pointers = discover_latest_completed_runs(zarr_path)
    by_family = {pointer.family_path: pointer for pointer in pointers}

    assert by_family["analysis/eye_angle_runs"].completion_status == "complete"
    assert by_family["analysis/tail_kinematics_runs"].completion_status == (
        "complete_via_parent_pointer"
    )
    assert by_family["keypoint_shard_runs"].rank_eligible is False


def test_build_storage_audit_report_ranks_runs_without_reading_values(
    tmp_path: Path,
) -> None:
    zarr_path = _make_archive(tmp_path / "recording_analysis.zarr")

    report = build_storage_audit_report(zarr_path)
    runs = {row["family_path"]: row for row in report["runs"]}

    eye = runs["analysis/eye_angle_runs"]
    assert eye["sharded_array_count"] == 0
    assert eye["wide_full_width_inner_chunk_array_count"] == 1
    assert eye["small_payload_file_count"] == 40
    assert eye["publication_provenance_status"] == "staged_publication_recorded"

    tail = runs["analysis/tail_kinematics_runs"]
    assert tail["sharded_array_count"] == 1
    assert tail["expected_payload_file_count"] == 40

    assert report["read_only_guard"]["unchanged"] is True
    assert report["read_only_guard"]["array_values_read"] is False
    assert all(
        not path.startswith("keypoint_shard_runs/")
        for path in report["rankings"]["by_physical_file_count"]
    )
    assert "Analysis storage candidate audit" in render_markdown(report)
