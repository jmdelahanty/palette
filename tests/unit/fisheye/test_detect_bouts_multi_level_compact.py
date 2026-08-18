from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.analysis.detect_bouts_multi_level as detect_bouts_multi_level
from fisheye.analysis.detect_bouts_multi_level import (
    PATH_DISTANCE_LEVEL_SOURCE,
    SPEED_LEVELS,
    SWIM_BOUT_STORED_LAYOUT_COMPACT_V2,
    _compute_global_metrics,
    _compute_inter_bout_intervals,
    _create_bout_points,
    _empty_peak_events,
    _store_detector_signal_matrix,
    _write_compact_v2_swim_bout_payloads,
    _bout_dtype,
)
from fisheye.analysis.swim_bout_io import load_default_swim_bout_tables
from fisheye.analysis.swim_bout_frame_axis import (
    FRAME_AXIS_CONTRACT_ATTR,
    FRAME_AXIS_STORAGE_EMBEDDED,
    build_frame_axis_contract,
)


def _one_bout(bout_id: int, start_frame: int, end_frame: int) -> np.ndarray:
    records = np.zeros(1, dtype=_bout_dtype())
    records["bout_id"] = bout_id
    records["start_frame"] = start_frame
    records["end_frame"] = end_frame
    records["core_start_frame"] = start_frame
    records["core_end_frame"] = end_frame
    records["duration_frames"] = end_frame - start_frame + 1
    records["duration_s"] = 0.1
    records["observed_duration_s"] = 0.1
    records["core_duration_s"] = 0.1
    records["path_length_mm"] = 1.5
    records["path_length_px"] = 15.0
    records["mean_speed_mm_s"] = 15.0
    records["peak_detection_signal_mm_s"] = 20.0
    records["peak_physical_speed_mm_s"] = 18.0
    records["valid_transition_fraction"] = 1.0
    records["start_time_s"] = start_frame / 60.0
    records["end_time_s"] = end_frame / 60.0
    records["core_start_time_s"] = start_frame / 60.0
    records["core_end_time_s"] = end_frame / 60.0
    return records


def test_detector_signal_matrix_uses_one_regular_chunk_without_transposing() -> None:
    root = zarr.group()
    signals = root.create_group("signals")
    frame_count = 8_193
    values = np.arange(frame_count, dtype=np.float32)[None, :]

    stored = _store_detector_signal_matrix(
        signals,
        "detector_signal_mm_s",
        values,
        attrs={
            "units": "mm/s",
            "axis_0": "detector_signal_id",
            "axis_1": "frame",
        },
    )

    assert stored.shape == (1, frame_count)
    assert stored.chunks == (1, frame_count)
    assert stored.shards is None
    assert stored.attrs["palette_physical_layout"] == "regular_chunks_v1"
    assert stored.attrs["palette_storage_schema_id"] == (
        "palette.swim_bout_detector_signal_storage.v2"
    )
    assert stored.attrs["palette_storage_policy"] == "single_regular_chunk_v1"
    assert stored.attrs["palette_shard_shape"] is None
    assert stored.attrs["palette_sharding_skip_reason"] == (
        "product_policy_single_regular_chunk"
    )
    assert stored.attrs["logical_axis_order"] == ["detector_signal_id", "frame"]
    np.testing.assert_array_equal(stored[0, :], values[0])


def test_peak_event_algorithm_contract_is_machine_interpretable() -> None:
    contract = detect_bouts_multi_level._swim_bout_algorithm_contract(
        method="peak_event",
        parameters={
            "min_bout_duration_s": 0.05,
            "resolved_min_bout_frames": 3,
            "boundary_mode": "threshold",
            "boundary_window_s": 0.25,
            "resolved_boundary_window_frames": 15,
            "min_peak_height_mm_s": None,
            "min_peak_prominence_mm_s": 4.0,
            "min_peak_distance_s": 0.1,
            "peak_width_rel_height": 0.98,
            "peak_event_boundary_mode": "relative_prominence_width",
            "shape_split_policy": "none",
        },
        source_track_path=(
            "analysis/track_kinematics_runs/offline/tk_run/tracks/id_0"
        ),
        track_id=0,
        fps=60.0,
        n_frames=120,
        speed_levels=list(SPEED_LEVELS),
        default_level="speed_exponential",
        path_distance_level_source={
            **PATH_DISTANCE_LEVEL_SOURCE,
            "speed_exponential": "filtered",
        },
        exponential_source_level="speed_filtered",
        exponential_tau_s=0.025,
        source_array_paths={
            "frame_indices": "resolved/track/frame_indices",
            "speed_mm": {
                "speed_filtered": "resolved/track/movement/speed/filtered/mm",
            },
            "frame_path_distance_mm": {
                "filtered": (
                    "resolved/track/movement/speed/filtered/frame_path_distance_mm"
                ),
            },
        },
    )

    assert contract["schema_id"] == "analysis.swim_bout_algorithm_contract"
    assert contract["schema_version"] == 1
    assert contract["active_detection"]["candidate_primitive"] == (
        "scipy.signal.find_peaks"
    )
    assert contract["active_detection"]["candidate_input_policy"] == (
        "replace every nonfinite value with 0.0 for peak operations"
    )
    assert contract["active_detection"]["gap_merge"] == "not_applied"
    assert "split at the finite minimum" in contract["active_detection"][
        "overlap_resolution"
    ]
    assert contract["signal_contracts"]["speed_exponential"]["transform"] == (
        "causal_exponential"
    )
    assert contract["causal_exponential_transform"]["source_path"] == (
        "resolved/track/movement/speed/filtered/mm"
    )
    assert contract["signal_contracts"]["speed_exponential"][
        "path_distance_source"
    ] == "resolved/track/movement/speed/filtered/frame_path_distance_mm"
    assert contract["persisted_metrics"]["mean_speed_mm_s"] == (
        "path_length_mm / observed_duration_s when defined"
    )


def test_source_path_contract_prefers_grouped_track_arrays() -> None:
    root = zarr.group()
    track = (
        root.create_group("analysis")
        .create_group("track_kinematics_runs")
        .create_group("offline")
        .create_group("tk_run")
        .create_group("tracks")
        .create_group("id_0")
    )
    track.create_array("frame_indices", data=np.arange(4, dtype=np.int64))
    track.create_array("speed_averaged_mm", data=np.arange(4, dtype=np.float32))
    filtered = (
        track.create_group("movement")
        .create_group("speed")
        .create_group("filtered")
    )
    filtered.create_array("mm", data=np.arange(4, dtype=np.float32))
    filtered.create_array(
        "frame_path_distance_mm",
        data=np.arange(4, dtype=np.float32),
    )

    paths = detect_bouts_multi_level._resolved_track_source_array_paths(
        SimpleNamespace(
            authority_status="verified_canonical_track_motion_v1",
            track_path=(
                "analysis/track_kinematics_runs/offline/tk_run/tracks/id_0"
            ),
            speed_mm_by_level={"filtered": object(), "averaged": object()},
            frame_path_distance_mm_by_level={"filtered": object()},
            delta_seconds=None,
            transition_valid=None,
            sample_valid=None,
            positions_mm=None,
            positions_px=None,
        ),
    )

    assert paths["speed_mm"]["speed_filtered"].endswith(
        "/movement/speed/filtered/mm"
    )
    assert paths["frame_path_distance_mm"]["filtered"].endswith(
        "/movement/speed/filtered/frame_path_distance_mm"
    )
    assert paths["speed_mm"]["speed_averaged"].endswith(
        "/movement/speed/averaged/mm"
    )


def test_compact_v2_writer_helper_outputs_resolver_readable_tables() -> None:
    root = zarr.group()
    analysis = root.create_group("analysis")
    parent = analysis.create_group("swim_bout_runs")
    parent.attrs["latest"] = "compact_run"
    parent.attrs["latest_complete"] = "compact_run"
    run = parent.create_group("compact_run")
    run.attrs.update(
        {
            "schema_id": "palette.swim_bout_runs",
            "schema_version": 7,
            "source_track_kinematics_run": "tk_run",
            "source_track_motion_manifest_sha256": "a" * 64,
            "track_id": 0,
            "default_level": "speed_exponential",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )

    speed_levels = list(SPEED_LEVELS)
    signal_id_by_level = {level: idx for idx, level in enumerate(speed_levels)}
    path_distance_source = {**PATH_DISTANCE_LEVEL_SOURCE, "speed_exponential": "filtered"}
    estimator_signal_id_by_level = {
        level: signal_id_by_level[f"speed_{path_distance_source[level]}"]
        for level in speed_levels
    }
    frames = np.arange(20, dtype=np.int64)
    frame_axis_contract = build_frame_axis_contract(
        frames,
        authoritative_path=(
            "analysis/track_kinematics_runs/offline/tk_run/tracks/id_0/"
            "source_acquisition_frame_index"
        ),
        source_track_kinematics_run="tk_run",
        track_id=0,
        source_track_motion_manifest_sha256="a" * 64,
        storage_mode=FRAME_AXIS_STORAGE_EMBEDDED,
    )
    run.attrs[FRAME_AXIS_CONTRACT_ATTR] = frame_axis_contract
    level_payloads = {}
    for idx, level in enumerate(speed_levels):
        bouts = _one_bout(idx + 1, idx * 2, idx * 2 + 1)
        intervals, _interval_metrics, hist = _compute_inter_bout_intervals(bouts, fps=60.0)
        level_payloads[level] = {
            "bouts": bouts,
            "peak_events": _empty_peak_events(),
            "intervals": intervals,
            "interval_histogram": hist,
            "global_metrics": _compute_global_metrics(bouts, fps=60.0, total_frames=20),
            "bout_points": _create_bout_points(bouts, None, None, frames, fps=60.0),
            "attrs": {"n_bouts": int(bouts.size), "speed_level": level},
        }

    _write_compact_v2_swim_bout_payloads(
        run,
        run_name="compact_run",
        speed_levels=speed_levels,
        level_payloads=level_payloads,
        signal_id_by_level=signal_id_by_level,
        estimator_signal_id_by_level=estimator_signal_id_by_level,
        default_level_key="speed_exponential",
        method="peak_event",
        parameters={
            "method": "peak_event",
            "boundary_mode": "threshold",
            "boundary_window_s": 0.25,
            "gap_merge_policy": "sampled_frame_gap",
            "min_bout_duration_s": 0.05,
            "min_gap_duration_s": 0.1,
            "min_gap_frames": None,
        },
        provenance={},
        track_id=0,
        pixel_to_mm=0.1,
        path_distance_level_source=path_distance_source,
        source_track_path="analysis/track_kinematics_runs/offline/tk_run/tracks/id_0",
        exponential_source_key="speed_filtered",
        exponential_tau_s=0.025,
        frames=frames,
        speeds={
            "speed_exponential_mm": np.linspace(0.0, 1.0, frames.size, dtype=np.float64),
        },
        frame_axis_contract=frame_axis_contract,
    )

    payload = load_default_swim_bout_tables(root, legacy_compatibility=True)

    assert run.attrs["layout"] == SWIM_BOUT_STORED_LAYOUT_COMPACT_V2
    assert "speed_exponential" not in run
    assert "indexes" in run
    assert "tables" in run
    detector_signal = run["signals/detector_signal_mm_s"]
    assert detector_signal.shape == (1, frames.size)
    assert detector_signal.chunks == (1, frames.size)
    assert detector_signal.shards is None
    assert detector_signal.attrs["axis_0"] == "detector_signal_id"
    assert detector_signal.attrs["axis_1"] == "frame"
    assert detector_signal.attrs["palette_storage_policy"] == (
        "single_regular_chunk_v1"
    )
    assert detector_signal.attrs["palette_sharding_skip_reason"] == (
        "product_policy_single_regular_chunk"
    )
    assert payload.signal.speed_level == "speed_exponential"
    assert payload.signal.role == "detector_response"
    assert payload.bouts["signal_id"].tolist() == [signal_id_by_level["speed_exponential"]]
    assert payload.bouts["estimator_signal_id"].tolist() == [signal_id_by_level["speed_filtered"]]
    assert payload.global_metrics["n_bouts"][0] == 1.0
    assert payload.series["detection_signal_mm_s"].shape == (frames.size,)


def test_detect_and_save_bouts_defaults_to_compact_v2_layout(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    frames = np.arange(12, dtype=np.int64)
    source_track = (
        root.create_group("analysis")
        .create_group("track_kinematics_runs")
        .create_group("offline")
        .create_group("tk_run")
        .create_group("tracks")
        .create_group("id_0")
    )
    source_track.create_array("frame_indices", data=frames)
    source_track.create_array("source_acquisition_frame_index", data=frames)
    speed = np.asarray([0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0], dtype=np.float64)
    transition_valid = np.ones(frames.size, dtype=bool)
    transition_valid[0] = False

    def fake_load_track(
        _zarr_path,
        _track_kinematics_run,
        track_id,
        *,
        track_kinematics_scope="offline",
    ):
        assert track_kinematics_scope == "offline"
        speeds = {
            "speed_raw_mm": speed,
            "speed_filtered_mm": speed,
            "speed_smoothed_mm": speed,
            "speed_averaged_mm": speed,
            "frames": frames,
            "frame_path_distance_raw_mm": speed / 60.0,
            "frame_path_distance_raw_px": speed / 6.0,
            "frame_path_distance_filtered_mm": speed / 60.0,
            "frame_path_distance_filtered_px": speed / 6.0,
            "frame_path_distance_smoothed_mm": speed / 60.0,
            "frame_path_distance_smoothed_px": speed / 6.0,
            "delta_seconds": np.full(frames.size, 1.0 / 60.0),
            "transition_valid": transition_valid,
            "sample_valid": np.ones(frames.size, dtype=bool),
        }
        metadata = {
            "fps": 60.0,
            "pixel_to_mm": 0.1,
            "n_frames": frames.size,
            "track_kinematics_run": "tk_run",
            "track_kinematics_scope": "offline",
            "source_track_path": (
                "analysis/track_kinematics_runs/offline/tk_run/tracks/id_0"
            ),
            "track_id": track_id,
            "positions_mm": np.column_stack((frames, frames)).astype(np.float64),
            "positions_px": np.column_stack((frames * 10, frames * 10)).astype(np.float64),
            "source_array_paths": {
                "frame_indices": (
                    "analysis/track_kinematics_runs/offline/tk_run/"
                    "tracks/id_0/frame_indices"
                ),
                "source_acquisition_frame_index": (
                    "analysis/track_kinematics_runs/offline/tk_run/"
                    "tracks/id_0/source_acquisition_frame_index"
                ),
            },
            "track_motion_authority": {
                "schema_id": "palette.track_motion_read_authority",
                "schema_version": 1,
                "run_ref": "/analysis/track_kinematics_runs/offline/tk_run",
                "track_ref": (
                    "/analysis/track_kinematics_runs/offline/tk_run/tracks/id_0"
                ),
                "track_id": 0,
                "motion_manifest_ref": (
                    "/analysis/track_kinematics_runs/offline/tk_run"
                    "@track_motion_publication_manifest"
                ),
                "motion_manifest_sha256": "a" * 64,
                "positions_px_ref": (
                    "/analysis/track_kinematics_runs/offline/tk_run/"
                    "tracks/id_0/positions_px"
                ),
                "positions_px_coordinate_descriptor_sha256": "b" * 64,
                "positions_mm_ref": (
                    "/analysis/track_kinematics_runs/offline/tk_run/"
                    "tracks/id_0/positions_mm"
                ),
                "positions_mm_coordinate_descriptor_sha256": "c" * 64,
                "track_sample_key_ref": (
                    "/analysis/track_kinematics_runs/offline/tk_run/"
                    "tracks/id_0/track_sample_key"
                ),
                "source_acquisition_frame_index_ref": (
                    "/analysis/track_kinematics_runs/offline/tk_run/"
                    "tracks/id_0/source_acquisition_frame_index"
                ),
            },
            "source_frame_indices_dtype": "int64",
            "source_frame_indices_shape": [frames.size],
        }
        return speeds, metadata

    monkeypatch.setattr(
        detect_bouts_multi_level,
        "_load_track_kinematics_track_speeds",
        fake_load_track,
    )

    detect_bouts_multi_level.detect_and_save_bouts(
        zarr_path=zarr_path,
        run_name="compact_run",
        track_kinematics_run="tk_run",
        method="threshold",
        threshold_mm=0.5,
        min_bout_duration_s=0.01,
        min_gap_duration_s=0.01,
        default_level="exponential",
        command="test",
    )

    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["analysis"]["swim_bout_runs"]["compact_run"]
    payload = load_default_swim_bout_tables(root)

    assert run.attrs["schema_version"] == 8
    assert run.attrs["layout"] == SWIM_BOUT_STORED_LAYOUT_COMPACT_V2
    assert "speed_exponential" not in run
    assert "frame_indices" not in run["signals"]
    frame_axis_contract = run.attrs[FRAME_AXIS_CONTRACT_ATTR]
    assert frame_axis_contract["storage_mode"] == "reference"
    assert frame_axis_contract["authoritative_path"] == (
        "analysis/track_kinematics_runs/offline/tk_run/tracks/id_0/"
        "source_acquisition_frame_index"
    )
    assert frame_axis_contract["shape"] == [frames.size]
    assert frame_axis_contract["authoritative_dtype"] == "int64"
    assert run.attrs["provenance"]["frame_axis_contract"] == frame_axis_contract
    assert run.attrs["source_track_motion_manifest_sha256"] == "a" * 64
    assert run.attrs["provenance"]["inputs"]["source_track_motion_authority"][
        "motion_manifest_sha256"
    ] == "a" * 64
    assert payload.signal.speed_level == "speed_exponential"
    assert payload.bouts.size > 0
    np.testing.assert_array_equal(payload.series["frame_indices"], frames)

    algorithm_contract = run.attrs["swim_bout_algorithm_contract"]
    assert algorithm_contract["schema_id"] == "analysis.swim_bout_algorithm_contract"
    assert algorithm_contract["schema_version"] == 1
    assert algorithm_contract["method_version"] == "detect_bouts_multi_level.v7"
    assert algorithm_contract["active_detection_method"] == "threshold"
    assert algorithm_contract["active_detection"]["candidate_rule"] == (
        "detection_signal_mm_s > threshold_mm"
    )
    assert algorithm_contract["causal_exponential_transform"]["source_level"] == (
        "speed_filtered"
    )
    assert algorithm_contract["causal_exponential_transform"]["recurrence"] == (
        "y[i] = alpha[i]*x[i] + (1-alpha[i])*y[i-1]"
    )
    assert algorithm_contract["boundaries"]["internal_interval_convention"] == (
        "half_open_sample_indices_[start,end_exclusive)"
    )
    assert run.attrs["provenance"]["algorithm_contract"] == algorithm_contract
    assert run.attrs["provenance"]["algorithm_contract_sha256"] == (
        run.attrs["swim_bout_algorithm_contract_sha256"]
    )

    phase_timing = run.attrs["phase_timing"]
    assert phase_timing["schema_id"] == "palette.swim_bout_phase_timing"
    assert phase_timing["schema_version"] == 1
    assert phase_timing["clock"] == "time.perf_counter"
    assert phase_timing["scope"] == "load_track_kinematics_through_payload_write"
    assert set(phase_timing["phase_durations_s"]) == {
        "load_track_kinematics",
        "build_exponential_response",
        "detect_levels",
        "initialize_output_and_metadata",
        "prepare_level_payloads",
        "write_payloads",
    }
    assert set(phase_timing["detection_levels"]) == {
        "speed_raw",
        "speed_filtered",
        "speed_smoothed",
        "speed_averaged",
        "speed_exponential",
    }
    assert all(
        elapsed_s >= 0.0
        for elapsed_s in phase_timing["phase_durations_s"].values()
    )
    assert all(
        level_timing["elapsed_s"] >= 0.0
        for level_timing in phase_timing["detection_levels"].values()
    )
    assert phase_timing["timed_pipeline_elapsed_s"] >= phase_timing["phase_sum_s"]
    assert phase_timing["unattributed_elapsed_s"] >= 0.0
    assert run.attrs["provenance"]["performance"] == phase_timing
