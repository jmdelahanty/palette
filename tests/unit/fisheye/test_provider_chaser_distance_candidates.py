from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.analysis.provider_chaser_distance_candidates as candidate_module
from fisheye.analysis.chaser_distance_runs import ChaserDistanceWindow
from fisheye.analysis.provider_chaser_distance_candidates import (
    MANIFEST_ATTR,
    ProviderChaserDistanceCandidate,
    ProviderChaserDistanceCandidateError,
    _controlled_run_path,
    _dense_fish_positions,
    _materialize_local,
    _stimulus_sample_positions,
    _summary,
    validate_provider_chaser_distance_candidate,
)


def _candidate(tmp_path: Path) -> ProviderChaserDistanceCandidate:
    frames = 4
    chasers = 2
    windows = (
        ChaserDistanceWindow(
            window_id=0,
            label="pre",
            start_frame=0,
            end_frame=1,
            start_time_s=0.0,
            end_time_s=0.1,
            duration_s=0.2,
        ),
        ChaserDistanceWindow(
            window_id=1,
            label="chaser",
            start_frame=2,
            end_frame=3,
            start_time_s=0.2,
            end_time_s=0.3,
            duration_s=0.2,
        ),
    )
    arrays = {
        "samples/stimulus_frame_num": np.arange(frames, dtype=np.int64),
        "samples/source_acquisition_frame_index": np.arange(frames, dtype=np.int64),
        "samples/timestamp_ns": np.arange(frames, dtype=np.int64) * 100_000_000,
        "samples/stimulus_epoch_window_id": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "samples/source_stimulus_run_row_index": np.arange(
            frames * chasers, dtype=np.int64
        ).reshape(frames, chasers),
        "samples/source_stimulus_source_row_index": np.arange(
            100, 100 + frames * chasers, dtype=np.int64
        ).reshape(frames, chasers),
        "positions/source_position_run_row_index": np.arange(frames, dtype=np.int64),
        "positions/source_position_source_row_index": np.arange(frames, dtype=np.int64),
        "positions/source_position_instance_key": np.arange(10, 14, dtype=np.uint64),
        "positions/source_position_failure_reason_code": np.zeros(frames, dtype=np.uint16),
        "positions/fish_position_source_camera_xy": np.asarray(
            [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0]],
            dtype=np.float32,
        ),
        "positions/fish_valid": np.ones(frames, dtype=bool),
        "positions/fish_position_arena_xy": np.asarray(
            [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0]],
            dtype=np.float32,
        ),
        "positions/chaser_position_arena_xy": np.asarray(
            [
                [[0.0, 0.0], [10.0, 0.0]],
                [[0.0, 0.0], [10.0, 0.0]],
                [[0.0, 0.0], [10.0, 0.0]],
                [[0.0, 0.0], [10.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        "positions/chaser_valid": np.ones((frames, chasers), dtype=bool),
        "chasers/chaser_index": np.asarray([0, 1], dtype=np.int16),
        "distances/distance_px": np.asarray(
            [[2.0, 9.0], [3.0, 8.0], [4.0, 7.0], [5.0, 6.0]],
            dtype=np.float32,
        ),
        "distances/distance_mm": np.asarray(
            [[1.0, 4.5], [1.5, 4.0], [2.0, 3.5], [2.5, 3.0]],
            dtype=np.float32,
        ),
        "distances/nearest_chaser_index": np.zeros(frames, dtype=np.int16),
        "distances/nearest_distance_mm": np.asarray(
            [1.0, 1.5, 2.0, 2.5], dtype=np.float32
        ),
        "epoch_summary/window_id": np.asarray([0, 1], dtype=np.int32),
        "epoch_summary/label_bytes": np.zeros((2, 96), dtype=np.uint8),
        "epoch_summary/start_frame": np.asarray([0, 2], dtype=np.int64),
        "epoch_summary/end_frame": np.asarray([1, 3], dtype=np.int64),
        "epoch_summary/valid_frame_count": np.full((2, 2), 2, dtype=np.int64),
        "epoch_summary/mean_distance_mm": np.ones((2, 2), dtype=np.float32),
        "epoch_summary/min_distance_mm": np.ones((2, 2), dtype=np.float32),
        "epoch_summary/p05_distance_mm": np.ones((2, 2), dtype=np.float32),
        "epoch_summary/p50_distance_mm": np.ones((2, 2), dtype=np.float32),
        "epoch_summary/p95_distance_mm": np.ones((2, 2), dtype=np.float32),
        "epoch_summary/fraction_within_threshold": np.ones((2, 2), dtype=np.float32),
        "epoch_distributions/window_id": np.asarray([0, 1], dtype=np.int32),
        "epoch_distributions/chaser_index": np.asarray([0, 1], dtype=np.int16),
        "epoch_distributions/bin_edges_mm": np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32),
        "epoch_distributions/bin_centers_mm": np.asarray([1.0, 3.0, 5.0], dtype=np.float32),
        "epoch_distributions/hist_counts": np.ones((2, 2, 3), dtype=np.uint32),
        "epoch_distributions/hist_density": np.full((2, 2, 3), 1 / 6, dtype=np.float32),
        "epoch_distributions/valid_sample_count": np.full((2, 2), 2, dtype=np.int64),
    }
    return ProviderChaserDistanceCandidate(
        source_zarr=tmp_path / "source.zarr",
        run_name="provider_canary_v1",
        recording_id="recording-1",
        position_run_path="analysis/subject_position_runs/observation/position_v1",
        position_manifest_sha256="a" * 64,
        position_estimator_id="keypoint_anatomical_triad_mean.v1",
        stimulus_run_path="analysis/stimulus_runs/stimulus_v1",
        stimulus_epoch_run_path="analysis/stimulus_epoch_runs/epochs_v1",
        total_frames=frames,
        fps=10.0,
        pixels_per_mm_projector=2.0,
        threshold_mm=20.0,
        distribution_bin_width_mm=2.0,
        windows=windows,
        arrays=arrays,
        source_authority=MappingProxyType(
            {
                "exact_test_fixture": True,
                "nested": MappingProxyType({"digest": "b" * 64}),
            }
        ),
    )


def test_exact_paths_reject_selector_and_noncanonical_spellings() -> None:
    assert (
        _controlled_run_path(
            "analysis/stimulus_runs/stimulus_v1",
            parent="analysis/stimulus_runs",
            label="stimulus",
        )
        == "analysis/stimulus_runs/stimulus_v1"
    )
    for value in (
        "analysis/stimulus_runs/latest",
        "analysis/stimulus_runs/stimulus_v1/child",
        "/analysis/stimulus_runs/stimulus_v1",
        "analysis/stimulus_runs/stimulus_v1/",
    ):
        with pytest.raises(ProviderChaserDistanceCandidateError):
            _controlled_run_path(
                value,
                parent="analysis/stimulus_runs",
                label="stimulus",
            )


def test_dense_provider_positions_preserve_exact_lineage_and_invalid_rows() -> None:
    result = _dense_fish_positions(
        total_frames=4,
        frames=np.asarray([0, 2, 3], dtype=np.int64),
        positions=np.asarray([[1.0, 2.0], [3.0, 4.0], [np.nan, 5.0]], dtype=np.float32),
        valid=np.asarray([True, False, True], dtype=bool),
        source_rows=np.asarray([100, 102, 103], dtype=np.int64),
        instance_keys=np.asarray([10, 12, 13], dtype=np.uint64),
        failure_reasons=np.asarray([0, 8, 9], dtype=np.uint16),
    )

    np.testing.assert_array_equal(result["fish_valid"], [True, False, False, False])
    np.testing.assert_array_equal(
        result["source_position_run_row_index"], [0, -1, 1, 2]
    )
    np.testing.assert_array_equal(
        result["source_position_source_row_index"], [100, -1, 102, 103]
    )
    assert np.isnan(result["fish_position_source_camera_xy"][1:]).all()
    assert result["source_position_failure_reason_code"].tolist() == [0, 65535, 8, 9]


def test_dense_provider_positions_fail_closed_on_duplicate_frame_lineage() -> None:
    with pytest.raises(
        ProviderChaserDistanceCandidateError,
        match="multiple rows for one acquisition frame",
    ):
        _dense_fish_positions(
            total_frames=2,
            frames=np.asarray([0, 0], dtype=np.int64),
            positions=np.zeros((2, 2), dtype=np.float32),
            valid=np.asarray([False, False], dtype=bool),
            source_rows=np.asarray([0, 1], dtype=np.int64),
            instance_keys=np.asarray([1, 2], dtype=np.uint64),
            failure_reasons=np.asarray([1, 1], dtype=np.uint16),
        )


def test_stimulus_samples_preserve_many_to_one_acquisition_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two 120 Hz stimulus samples legitimately bind acquisition frame 0 from
    # the 100 fps camera. Both samples and both chasers must survive.
    stimulus_frames = np.repeat(np.asarray([10, 11, 12], dtype=np.int64), 2)
    chaser_indices = np.tile(np.asarray([0, 1], dtype=np.int64), 3)
    monkeypatch.setattr(
        candidate_module,
        "_identity_component",
        lambda _stimulus, name: (
            stimulus_frames if name == "stimulus_frame_num" else chaser_indices
        ),
    )
    group = {
        "chaser_position_xy": np.asarray(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [1.0, 0.0],
                [9.0, 0.0],
                [2.0, 0.0],
                [8.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "timestamp_ns_session": np.repeat(
            np.asarray([0, 8_333_333, 16_666_666], dtype=np.int64), 2
        ),
        "source_row_indices": np.arange(100, 106, dtype=np.int64),
    }
    stimulus = SimpleNamespace(
        source_acquisition_frame_index=np.repeat(
            np.asarray([0, 0, 1], dtype=np.int64), 2
        )
    )

    samples = _stimulus_sample_positions(stimulus, group, total_frames=2)

    np.testing.assert_array_equal(samples["stimulus_frame_num"], [10, 11, 12])
    np.testing.assert_array_equal(
        samples["source_acquisition_frame_index"], [0, 0, 1]
    )
    assert samples["chaser_position_arena_xy"].shape == (3, 2, 2)
    np.testing.assert_array_equal(
        samples["source_stimulus_source_row_index"],
        [[100, 101], [102, 103], [104, 105]],
    )


def test_local_candidate_is_ineligible_and_uses_provider_lineage(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    local_zarr = tmp_path / "candidate.zarr"
    run_path, manifest_sha = _materialize_local(candidate, local_zarr=local_zarr)

    direct = validate_provider_chaser_distance_candidate(
        run_path,
        use_consolidated=False,
        expected_manifest_sha256=manifest_sha,
    )
    consolidated = validate_provider_chaser_distance_candidate(
        run_path,
        use_consolidated=True,
        archive_path=local_zarr,
        archive_run_path=candidate.run_path,
        expected_manifest_sha256=manifest_sha,
    )
    assert direct["valid"] is True
    assert consolidated["valid"] is True

    run = zarr.open_group(str(run_path), mode="r", zarr_format=3, use_consolidated=False)
    assert run.attrs["stage_selector_eligible"] is False
    assert run["visualizations"].attrs["distance_histogram_semantics"] == (
        "fraction_of_valid_stimulus_samples_per_shared_linear_distance_bin"
    )
    manifest_paths = {item["path"] for item in run.attrs[MANIFEST_ATTR]["payload"]["arrays"]}
    assert "positions/source_position_run_row_index" in manifest_paths
    assert not any("detection" in path for path in manifest_paths)
    assert _summary(candidate)["fish_valid_sample_count"] == 4


def test_candidate_validation_detects_payload_tampering(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    run_path, manifest_sha = _materialize_local(
        candidate,
        local_zarr=tmp_path / "candidate.zarr",
    )
    run = zarr.open_group(str(run_path), mode="a", zarr_format=3, use_consolidated=False)
    run["distances/nearest_distance_mm"][0] = np.float32(99.0)

    result = validate_provider_chaser_distance_candidate(
        run_path,
        use_consolidated=False,
        expected_manifest_sha256=manifest_sha,
    )
    assert result["valid"] is False
    assert "array mismatch:distances/nearest_distance_mm" in result["errors"]
