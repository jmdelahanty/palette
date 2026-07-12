from __future__ import annotations

import json
from pathlib import Path

import zarr

from fisheye.registry.db import Registry
from fisheye.registry.extractors.stimulus_metadata import extract_stimulus_metadata


def _write_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "mixed_analysis.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    root.attrs.update(
        {
            "dataset_id": "dataset_mixed",
            "recording_id": "mixed",
            "session_uuid": "mixed",
            "zarr_purpose": "analysis",
            "zarr_use": "analysis",
            "recording_name": "mixed",
            "recording_type": "behavior",
        }
    )
    parent = root.require_group("analysis/stimulus_runs")
    run = parent.create_group("stimulus_001")
    run.attrs["protocol_json"] = json.dumps(
        {
            "protocol_name": "Mixed canary",
            "steps": [
                {
                    "name": "grating one",
                    "duration_s": 5.0,
                    "parameters": {"stimulus_mode": "MOVING_GRATING"},
                },
                {
                    "name": "grating two",
                    "duration_s": 7.0,
                    "parameters": {"stimulus_mode": "MOVING_GRATING"},
                },
                {
                    "name": "loom",
                    "duration_s": 3.0,
                    "parameters": {"stimulus_mode": "LOOMING_DOT"},
                },
            ],
        }
    )
    steps = run.create_group("steps")
    for index, (mode, duration) in enumerate(
        (("MOVING_GRATING", 5.0), ("MOVING_GRATING", 7.0), ("LOOMING_DOT", 3.0))
    ):
        step = steps.create_group(f"step_{index}")
        step.attrs.update(
            {
                "step_index": index,
                "step_name": f"step {index}",
                "stimulus_mode": mode,
                "start_camera_frame": index * 100,
                "end_camera_frame": (index + 1) * 100,
                "duration_s": duration,
            }
        )
    parent.attrs["latest"] = "stimulus_001"
    return zarr_path


def test_extract_stimulus_metadata_normalizes_steps_and_mode_counts(tmp_path: Path) -> None:
    zarr_path = _write_archive(tmp_path)
    root = zarr.open_group(zarr_path, mode="r", use_consolidated=False)

    result = extract_stimulus_metadata(
        root,
        zarr_path=zarr_path,
        recording_id="mixed",
    )

    assert len(result.protocols) == 1
    assert result.protocols[0]["protocol_name"] == "Mixed canary"
    assert len(result.protocol_steps) == 3
    assert len(result.recording_steps) == 3
    assert [
        (row["stimulus_mode"], row["step_count"], row["total_duration_s"])
        for row in result.recording_modes
    ] == [("LOOMING_DOT", 1, 3.0), ("MOVING_GRATING", 2, 12.0)]


def test_registry_scan_exposes_protocol_steps_and_recording_mode_counts(
    tmp_path: Path,
) -> None:
    zarr_path = _write_archive(tmp_path)
    registry = Registry(tmp_path / "registry.sqlite")
    try:
        dataset_id = registry.scan_zarr(zarr_path)
        protocol = registry.conn.execute(
            "SELECT protocol_name, step_count FROM stimulus_protocols"
        ).fetchone()
        steps = registry.conn.execute(
            """
            SELECT step_index, stimulus_mode
            FROM recording_stimulus_steps
            WHERE dataset_id = ?
            ORDER BY step_index
            """,
            (dataset_id,),
        ).fetchall()
        modes = registry.conn.execute(
            """
            SELECT stimulus_mode, step_count, total_duration_s
            FROM recording_stimulus_mode_counts
            WHERE dataset_id = ? AND is_latest = 1
            ORDER BY stimulus_mode
            """,
            (dataset_id,),
        ).fetchall()
    finally:
        registry.close()

    assert protocol is not None
    assert tuple(protocol) == ("Mixed canary", 3)
    assert [tuple(row) for row in steps] == [
        (0, "MOVING_GRATING"),
        (1, "MOVING_GRATING"),
        (2, "LOOMING_DOT"),
    ]
    assert [tuple(row) for row in modes] == [
        ("LOOMING_DOT", 1, 3.0),
        ("MOVING_GRATING", 2, 12.0),
    ]
