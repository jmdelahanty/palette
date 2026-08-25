from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import zarr

from fisheye.analysis.import_stimulus_to_zarr import backfill_stimulus_step_metadata
from fisheye.utils import backfill_stimulus_step_metadata as cli


def _write_stimulus_h5(path: Path) -> None:
    metadata_dtype = np.dtype(
        [
            ("stimulus_frame_num", np.uint64),
            ("triggering_camera_frame_id", np.uint64),
            ("timestamp_ns", np.int64),
        ]
    )
    frame_metadata = np.array(
        [
            (0, 10, 0),
            (60, 70, 1_000_000_000),
            (120, 130, 2_000_000_000),
        ],
        dtype=metadata_dtype,
    )
    events_dtype = np.dtype(
        [
            ("event_name", "S32"),
            ("current_step_index", np.int32),
            ("stimulus_mode_id", np.int32),
            ("camera_frame_id", np.int64),
        ]
    )
    events = np.array(
        [
            (b"STEP_START", 0, 3, 10),
            (b"STEP_END", 0, 3, 70),
            (b"STEP_START", 1, 6, 80),
            (b"STEP_END", 1, 6, 140),
        ],
        dtype=events_dtype,
    )
    arena_config = {
        "active_camera_id": "2010093",
        "experimental_area_center_x_px": 172.0,
        "experimental_area_center_y_px": 172.0,
        "experimental_area_radius_mm": 40.0,
        "camera_calibrations": [
            {
                "camera_id": "2010093",
                "pixels_per_mm_camera": 50.0,
                "pixels_per_mm_projector": 5.0,
            }
        ],
    }
    protocol = {
        "steps": [
            {
                "name": "left grating",
                "stimulus_mode_str": "MOVING_GRATING",
                "duration_seconds": 1.0,
                "parameters": {
                    "type": "ProtocolMovingGratingParams",
                    "orientation_degrees": 180.0,
                    "speed_mm_per_sec": 3.5,
                    "speed_pps": 17.5,
                    "spatial_freq_cycles_per_mm": 0.2,
                    "spatial_freq_rpp": 0.04,
                },
            },
            {
                "name": "concentric center",
                "stimulus_mode_str": "CONCENTRIC_GRATING",
                "duration_seconds": 1.0,
                "parameters": {
                    "type": "ProtocolConcentricGratingParams",
                    "is_expanding": False,
                    "speed_mm_per_sec": 4.0,
                    "speed_pps": 20.0,
                    "spatial_freq_cycles_per_mm": 0.25,
                    "spatial_freq_rpp": 0.05,
                    "stimulus_role": "centering_utility",
                    "target_radius_min_mm": 8.0,
                    "target_radius_max_mm": 14.0,
                },
            },
        ]
    }

    with h5py.File(path, "w") as h5:
        video = h5.create_group("video_metadata")
        video.create_dataset("frame_metadata", data=frame_metadata)
        h5.create_dataset("events", data=events)

        protocol_group = h5.create_group("protocol_snapshot")
        protocol_group.create_dataset(
            "protocol_definition_json",
            data=json.dumps(protocol).encode("utf-8"),
        )

        calib = h5.create_group("calibration_snapshot")
        calib.create_dataset("arena_config_json", data=json.dumps(arena_config).encode("utf-8"))
        cam = calib.create_group("2010093")
        cam.attrs["pixels_per_mm_projector"] = 5.0

        coords = h5.create_group("stimulus_coordinates")
        arena = coords.create_group("arena_1")
        custom = arena.create_group("custom_coordinates")
        custom.attrs["texture_center_x"] = 172.0
        custom.attrs["texture_center_y"] = 173.0


def _seed_zarr(
    zarr_path: Path,
    *,
    source_h5: Path | None,
    published: bool = False,
) -> None:
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs["zarr_purpose"] = "analysis"
    runs = root.create_group("analysis").create_group("stimulus_runs")
    run = runs.create_group("stimulus_001")
    if source_h5 is not None:
        run.attrs["source_h5"] = str(source_h5)
    if published:
        runs.attrs["latest"] = "stimulus_001"


def _add_modern_semantic_snapshot(path: Path) -> str:
    semantic = {
        "identity": {
            "iti_stimulus_mode_id": 99,
            "steps": [
                {
                    "duration": {"scale": "1e-3", "unit": "s", "value": 1000},
                    "parameters": {},
                    "post_stimulus_iti": {
                        "scale": "1e-3",
                        "unit": "s",
                        "value": 0,
                    },
                    "stimulus_mode_id": 3,
                },
                {
                    "duration": {"scale": "1e-3", "unit": "s", "value": 1000},
                    "parameters": {},
                    "post_stimulus_iti": {
                        "scale": "1e-3",
                        "unit": "s",
                        "value": 0,
                    },
                    "stimulus_mode_id": 6,
                },
            ],
        },
        "normalization_policy": "citrus.protocol.semantic.v1",
        "schema_id": "citrus.protocol.semantic",
        "schema_version": 1,
    }
    semantic_json = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
    semantic_hash = "sha256:" + sha256(semantic_json.encode("utf-8")).hexdigest()
    trial = {
        "normalization_policy": "citrus.protocol.trial_index.v1",
        "protocol_semantic_hash": semantic_hash,
        "schema_id": "citrus.protocol.trial_index",
        "schema_version": 1,
        "steps": [
            {
                "duration_s": 1.0,
                "features": {},
                "index_status": "detailed",
                "post_stimulus_iti_s": 0.0,
                "step_index": 0,
                "stimulus_family": "moving_grating",
                "stimulus_mode": "MOVING_GRATING",
                "stimulus_mode_id": 3,
            },
            {
                "duration_s": 1.0,
                "features": {},
                "index_status": "detailed",
                "post_stimulus_iti_s": 0.0,
                "step_index": 1,
                "stimulus_family": "concentric_grating",
                "stimulus_mode": "CONCENTRIC_GRATING",
                "stimulus_mode_id": 6,
            },
        ],
    }
    trial_json = json.dumps(trial, sort_keys=True, separators=(",", ":"))
    with h5py.File(path, "r+") as h5:
        group = h5["protocol_snapshot"]
        group.create_dataset("protocol_semantic_hash", data=semantic_hash.encode("utf-8"))
        group.create_dataset("protocol_semantic_json", data=semantic_json.encode("utf-8"))
        group.create_dataset("protocol_trial_index_json", data=trial_json.encode("utf-8"))
    return semantic_hash


def test_backfill_stimulus_step_metadata_dry_run_does_not_write(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_zarr(zarr_path, source_h5=h5_path)

    summary = backfill_stimulus_step_metadata(zarr_path, apply=False)

    assert summary["runs_scanned"] == 1
    assert summary["details"][0]["status"] == "would_backfill"
    assert summary["details"][0]["step_count"] == 2
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["stimulus_runs"]["stimulus_001"]
    assert "steps" not in run
    assert "stimulus_coordinates" not in run


def test_backfill_stimulus_step_metadata_writes_steps_and_coordinates(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_zarr(zarr_path, source_h5=h5_path)

    summary = backfill_stimulus_step_metadata(zarr_path, apply=True)

    assert summary["details"][0]["status"] == "backfilled"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["analysis"]["stimulus_runs"]["stimulus_001"]
    assert "protocol_json" in run.attrs
    assert run.attrs["protocol_semantic_status"] == "legacy_missing"

    step0 = run["steps"]["step_0"]
    assert step0.attrs["step_name"] == "left grating"
    assert step0.attrs["stimulus_mode"] == "MOVING_GRATING"
    assert step0["moving_grating"].attrs["grating_direction_camera_deg"] == 180.0

    step1 = run["steps"]["step_1"]
    concentric = step1["concentric_grating"]
    assert concentric.attrs["radial_polarity_authored"] == "contracting"
    assert concentric.attrs["stimulus_role"] == "centering_utility"
    assert concentric.attrs["coordinate_geometry_status"] == (
        "not_materialized_missing_exact_protocol_coordinate_contract"
    )
    for attr_name in (
        "center_x_px",
        "center_y_px",
        "center_coordinate_frame",
        "center_source",
        "center_x_mm",
        "center_y_mm",
        "center_mm_source",
        "center_mm_coordinate_frame",
        "pixels_per_mm_projector",
    ):
        assert attr_name not in concentric.attrs
    assert concentric.attrs["target_radius_min_mm"] == 8.0
    assert run["stimulus_coordinates"]["arena_1"]["custom_coordinates"].attrs["texture_center_y"] == 173.0


def test_backfill_stimulus_step_metadata_applies_configured_grating_offset(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_zarr(zarr_path, source_h5=h5_path)

    summary = backfill_stimulus_step_metadata(
        zarr_path,
        apply=True,
        moving_grating_camera_offset_deg=180.0,
    )

    assert summary["details"][0]["status"] == "backfilled"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    moving = root["analysis"]["stimulus_runs"]["stimulus_001"]["steps"]["step_0"]["moving_grating"]
    assert moving.attrs["orientation_degrees_authored"] == 180.0
    assert moving.attrs["grating_direction_camera_deg"] == 0.0
    assert moving.attrs["camera_to_projector_offset_deg"] == 180.0
    assert moving.attrs["direction_mapping_status"] == "configured_camera_offset"
    assert moving.attrs["direction_mapping_validated"] is False


def test_backfill_stimulus_step_metadata_skips_existing_unless_overwrite(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_zarr(zarr_path, source_h5=h5_path)

    assert backfill_stimulus_step_metadata(zarr_path, apply=True)["details"][0]["status"] == "backfilled"
    assert backfill_stimulus_step_metadata(zarr_path, apply=False)["details"][0]["status"] == "skipped_existing"
    assert (
        backfill_stimulus_step_metadata(zarr_path, apply=False, overwrite=True)["details"][0]["status"]
        == "would_overwrite"
    )


def test_cli_backfill_falls_back_to_recording_raw_h5(tmp_path: Path) -> None:
    recording = tmp_path / "2026-01-01T00-00-00Z_arena_1"
    raw_dir = recording / "raw"
    zarr_dir = recording / "zarr"
    raw_dir.mkdir(parents=True)
    zarr_dir.mkdir()
    h5_path = raw_dir / "2026-01-01T00-00-00Z_arena_1.h5"
    zarr_path = zarr_dir / "2026-01-01T00-00-00Z_arena_1_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_zarr(zarr_path, source_h5=None)

    rc = cli.main([str(recording), "--recursive", "--apply"])

    assert rc == 0
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert root["analysis"]["stimulus_runs"]["stimulus_001"]["steps"]["step_1"].attrs["stimulus_mode"] == (
        "CONCENTRIC_GRATING"
    )


def test_published_run_requires_immutable_successor(tmp_path: Path) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _seed_zarr(zarr_path, source_h5=h5_path, published=True)

    summary = backfill_stimulus_step_metadata(zarr_path, apply=True)

    assert summary["details"][0]["status"] == "requires_immutable_successor"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["analysis"]["stimulus_runs"]["stimulus_001"]
    assert "steps" not in run
    assert "protocol_semantic_status" not in run.attrs


def test_modern_backfill_reveals_and_materializes_exact_semantic_identity(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    semantic_hash = _add_modern_semantic_snapshot(h5_path)
    _seed_zarr(zarr_path, source_h5=h5_path)

    dry_run = backfill_stimulus_step_metadata(zarr_path, apply=False)
    detail = dry_run["details"][0]
    assert detail["status"] == "would_backfill"
    assert detail["protocol_semantic_hash"] == semantic_hash
    assert detail["protocol_recipe_label"] == (
        "MOVING_GRATING -> CONCENTRIC_GRATING"
    )
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "protocol_semantic_snapshot" not in root["analysis"]["stimulus_runs"]["stimulus_001"]

    applied = backfill_stimulus_step_metadata(zarr_path, apply=True)
    assert applied["details"][0]["status"] == "backfilled"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["analysis"]["stimulus_runs"]["stimulus_001"]
    assert run.attrs["protocol_semantic_status"] == "verified"
    assert run.attrs["protocol_semantic_hash"] == semantic_hash
    assert run["steps"]["step_0"].attrs["protocol_semantic_step_index"] == 0
    assert run["steps"]["step_1"].attrs["stimulus_family"] == (
        "concentric_grating"
    )
    stored = run["protocol_semantic_snapshot"]
    with h5py.File(h5_path, "r") as h5:
        assert np.asarray(stored["protocol_semantic_json_utf8"][:]).tobytes() == h5[
            "/protocol_snapshot/protocol_semantic_json"
        ][()]
        assert np.asarray(stored["protocol_trial_index_json_utf8"][:]).tobytes() == h5[
            "/protocol_snapshot/protocol_trial_index_json"
        ][()]

    repeated = backfill_stimulus_step_metadata(zarr_path, apply=True)
    assert repeated["details"][0]["status"] == "skipped_existing"


def test_modern_backfill_binds_existing_steps_without_rewriting_them(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    _add_modern_semantic_snapshot(h5_path)
    _seed_zarr(zarr_path, source_h5=h5_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    run = root["analysis"]["stimulus_runs"]["stimulus_001"]
    run.attrs["protocol_json"] = "{}"
    run.create_group("stimulus_coordinates")
    steps = run.create_group("steps")
    for index, (mode_id, mode_name) in enumerate(
        ((3, "MOVING_GRATING"), (6, "CONCENTRIC_GRATING"))
    ):
        step = steps.create_group(f"step_{index}")
        step.attrs.update(
            {
                "step_index": index,
                "stimulus_mode_id": mode_id,
                "stimulus_mode": mode_name,
                "duration_s": 1.0,
                "sentinel": f"preserve-{index}",
            }
        )

    summary = backfill_stimulus_step_metadata(zarr_path, apply=True)

    assert summary["details"][0]["bound_existing_steps"] is True
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    rebound = root["analysis"]["stimulus_runs"]["stimulus_001"]["steps"]
    assert rebound["step_0"].attrs["sentinel"] == "preserve-0"
    assert rebound["step_1"].attrs["sentinel"] == "preserve-1"


def test_modern_backfill_rejects_corrupt_stored_trial_index_without_mutation(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "session.h5"
    zarr_path = tmp_path / "sample_analysis.zarr"
    _write_stimulus_h5(h5_path)
    semantic_hash = _add_modern_semantic_snapshot(h5_path)
    _seed_zarr(zarr_path, source_h5=h5_path)
    applied = backfill_stimulus_step_metadata(zarr_path, apply=True)
    assert applied["details"][0]["status"] == "backfilled"

    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    run = root["analysis/stimulus_runs/stimulus_001"]
    stored_trial = run[
        "protocol_semantic_snapshot/protocol_trial_index_json_utf8"
    ]
    original_first_byte = int(stored_trial[0])
    stored_trial[0] = ord("[")

    summary = backfill_stimulus_step_metadata(zarr_path, apply=True)

    assert summary["details"][0]["status"] == "error"
    assert "differ from the exact source H5" in summary["details"][0]["reason"]
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root["analysis/stimulus_runs/stimulus_001"]
    assert run.attrs["protocol_semantic_status"] == "verified"
    assert run.attrs["protocol_semantic_hash"] == semantic_hash
    assert int(
        run["protocol_semantic_snapshot/protocol_trial_index_json_utf8"][0]
    ) == ord("[")
    assert original_first_byte != ord("[")
