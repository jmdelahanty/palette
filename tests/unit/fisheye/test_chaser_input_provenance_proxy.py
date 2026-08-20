from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pytest

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    BEHAVIORAL_DENOMINATOR,
    CAMERA_EXPOSURE_REFERENCE,
    CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE,
    PHYSICAL_PRESENTATION_VERIFIED,
    PRESENTATION_TIMESTAMP_AVAILABLE,
    PROXY_POLICY_ID,
    PROJECTION_RECORD_SCHEMA_ID,
    PROJECTION_RECORD_SCHEMA_VERSION,
    SCIENTIFIC_USE_CLASS,
    TEMPORAL_ALIGNMENT_CLASS,
    TEMPORAL_ALIGNMENT_REQUIREMENT,
    ChaserInputProvenanceProxyError,
    select_chaser_input_provenance_proxy,
)


@dataclass(frozen=True, slots=True)
class _Dimensions:
    total_frames: int


@dataclass(frozen=True, slots=True)
class _Source:
    recording_id: str
    dimensions: _Dimensions
    stimulus_frame_num: np.ndarray
    timestamp_ns: np.ndarray
    source_acquisition_frame_index: np.ndarray
    source_sample_row_index: np.ndarray
    source_stimulus_run_row_index: np.ndarray
    source_stimulus_source_row_index: np.ndarray
    chaser_index: np.ndarray
    chaser_position_arena_xy: np.ndarray
    chaser_valid: np.ndarray
    authorities: MappingProxyType
    run_path: str = "analysis/provider_chaser_distance_candidate_runs/native"
    manifest_sha256: str = "a" * 64
    verification_digest: str = "b" * 64

    def assert_verified(self) -> None:
        return None


def _source(
    *,
    acquisition: list[int] | None = None,
    timestamps: list[int] | None = None,
    stimulus_frames: list[int] | None = None,
    chaser_valid: np.ndarray | None = None,
    chaser_positions: np.ndarray | None = None,
    sample_rows: list[int] | None = None,
    nested_recording_id: str = "recording-1",
) -> _Source:
    n_samples = len(acquisition or [0, 0, 0, 2, 4])
    n_chasers = 2
    if stimulus_frames is None:
        stimulus_frames = list(range(10, 10 + n_samples))
    if timestamps is None:
        timestamps = [100, 200, 200, 300, 400][:n_samples]
    if acquisition is None:
        acquisition = [0, 0, 0, 2, 4][:n_samples]
    if sample_rows is None:
        sample_rows = list(range(n_samples))
    if chaser_valid is None:
        chaser_valid = np.ones((n_samples, n_chasers), dtype=bool)
    if chaser_positions is None:
        chaser_positions = np.asarray(
            [
                [[10.0, 100.0], [11.0, 101.0]],
                [[20.0, 200.0], [21.0, 201.0]],
                [[30.0, 300.0], [31.0, 301.0]],
                [[40.0, 400.0], [41.0, 401.0]],
                [[50.0, 500.0], [51.0, 501.0]],
            ][:n_samples],
            dtype=np.float64,
        )
    source_rows = np.arange(n_samples * n_chasers, dtype=np.int64).reshape(
        n_samples, n_chasers
    )
    authority = MappingProxyType(
        {
            "schema_id": "native-source-authority-v1",
            "recording_id": nested_recording_id,
            "position": MappingProxyType({"recording_id": nested_recording_id}),
            "stimulus": MappingProxyType({"recording_id": nested_recording_id}),
        }
    )
    return _Source(
        recording_id="recording-1",
        dimensions=_Dimensions(total_frames=8),
        stimulus_frame_num=np.asarray(stimulus_frames, dtype=np.int64),
        timestamp_ns=np.asarray(timestamps, dtype=np.int64),
        source_acquisition_frame_index=np.asarray(acquisition, dtype=np.int64),
        source_sample_row_index=np.asarray(sample_rows, dtype=np.int64),
        source_stimulus_run_row_index=source_rows,
        source_stimulus_source_row_index=source_rows + 100,
        chaser_index=np.asarray([0, 1], dtype=np.int16),
        chaser_position_arena_xy=chaser_positions,
        chaser_valid=chaser_valid,
        authorities=authority,
    )


def test_selects_one_complete_same_sample_using_timestamp_then_stimulus_frame() -> None:
    source = _source()
    result = select_chaser_input_provenance_proxy(source)

    np.testing.assert_array_equal(result.acquisition_frame_index, [0, 2, 4])
    np.testing.assert_array_equal(result.candidate_sample_count, [3, 1, 1])
    np.testing.assert_array_equal(result.candidate_offsets, [0, 3, 4, 5])
    np.testing.assert_array_equal(result.selected, [True, True, True])
    np.testing.assert_array_equal(result.selected_native_sample_row_index, [2, 3, 4])
    # Row 2 supplies both chasers.  The selector cannot take chaser 0 from
    # row 2 and chaser 1 from row 1.
    np.testing.assert_array_equal(
        result.selected_chaser_position_xy[0],
        [[30.0, 300.0], [31.0, 301.0]],
    )
    np.testing.assert_array_equal(result.candidate_native_sample_row_index, [0, 1, 2, 3, 4])
    assert result.provenance["policy_id"] == PROXY_POLICY_ID
    assert result.provenance["temporal_alignment_class"] == TEMPORAL_ALIGNMENT_CLASS


def test_missing_acquisition_frames_are_not_carried_forward() -> None:
    result = select_chaser_input_provenance_proxy(_source())

    np.testing.assert_array_equal(result.acquisition_frame_index, [0, 2, 4])
    assert 1 not in result.acquisition_frame_index.tolist()
    assert result.unique_acquisition_frame_count == 3
    assert result.behavioral_denominator == BEHAVIORAL_DENOMINATOR
    assert result.provenance["no_carry_forward_across_missing_acquisition_frames"] is True


def test_incomplete_sample_is_retained_but_fails_closed_for_that_frame() -> None:
    valid = np.ones((5, 2), dtype=bool)
    valid[4, 1] = False
    result = select_chaser_input_provenance_proxy(_source(chaser_valid=valid))

    np.testing.assert_array_equal(result.acquisition_frame_index, [0, 2, 4])
    np.testing.assert_array_equal(result.selected, [True, True, False])
    assert result.selection_reason_code[2] == "no_complete_chaser_sample"
    assert result.candidate_reason_code[4] == "incomplete_chaser_sample"
    np.testing.assert_array_equal(result.selected_chaser_valid[2], [False, False])
    assert np.isnan(result.selected_chaser_position_xy[2]).all()


def test_nonfinite_coordinate_marked_valid_fails_closed() -> None:
    positions = _source().chaser_position_arena_xy.copy()
    positions[1, 0, 0] = np.nan
    with pytest.raises(ChaserInputProvenanceProxyError, match="marked valid"):
        select_chaser_input_provenance_proxy(_source(chaser_positions=positions))


@pytest.mark.parametrize(
    ("field", "values", "message"),
    [
        ("timestamps", [100, 50, 200, 300, 400], "timestamp"),
        ("acquisition", [0, 2, 1, 2, 4], "acquisition_frame_index"),
        ("stimulus_frames", [10, 12, 12, 13, 14], "stimulus_frame_num"),
        ("sample_rows", [0, 2, 1, 3, 4], "sample rows"),
    ],
)
def test_malformed_native_ordering_fails_closed(
    field: str, values: list[int], message: str
) -> None:
    kwargs = {field: values}
    with pytest.raises(ChaserInputProvenanceProxyError, match=message):
        select_chaser_input_provenance_proxy(_source(**kwargs))


def test_mixed_recording_authority_fails_closed() -> None:
    with pytest.raises(ChaserInputProvenanceProxyError, match="mixed recording"):
        select_chaser_input_provenance_proxy(
            _source(nested_recording_id="different-recording")
        )


def test_unverified_structural_source_is_rejected() -> None:
    source = _source()

    class _Unverified:
        pass

    unverified = _Unverified()
    for field in source.__dataclass_fields__:
        setattr(unverified, field, getattr(source, field))

    with pytest.raises(ChaserInputProvenanceProxyError, match="assert_verified"):
        select_chaser_input_provenance_proxy(unverified)


def test_output_arrays_are_read_only_copies_and_source_is_unchanged() -> None:
    source = _source()
    original = source.chaser_position_arena_xy.copy()
    result = select_chaser_input_provenance_proxy(source)

    assert result.selected_chaser_position_xy.flags.writeable is False
    assert result.candidate_offsets.flags.writeable is False
    with pytest.raises(ValueError):
        result.selected_chaser_position_xy[0, 0, 0] = 999.0
    np.testing.assert_array_equal(source.chaser_position_arena_xy, original)
    assert result.source_handle is source


def test_exact_tie_breaker_is_deterministic_by_source_sample_row() -> None:
    source = _source(
        stimulus_frames=[10, 11, 12, 13, 14],
        timestamps=[100, 100, 100, 300, 400],
        sample_rows=[0, 1, 2, 3, 4],
    )
    result = select_chaser_input_provenance_proxy(source)
    # The stimulus frame is the second tie-breaker, so row 2 wins the frame-0
    # tie even though all three timestamps are identical.
    assert result.selected_native_sample_row_index[0] == 2


def test_provenance_constants_are_exact() -> None:
    result = select_chaser_input_provenance_proxy(_source())
    provenance = result.provenance
    assert provenance["physical_presentation_verified"] is PHYSICAL_PRESENTATION_VERIFIED
    assert provenance["presentation_timestamp_available"] is PRESENTATION_TIMESTAMP_AVAILABLE
    assert (
        provenance["camera_presentation_clock_transform_available"]
        is CAMERA_PRESENTATION_CLOCK_TRANSFORM_AVAILABLE
    )
    assert provenance["camera_exposure_reference"] == CAMERA_EXPOSURE_REFERENCE
    assert provenance["scientific_use_class"] == SCIENTIFIC_USE_CLASS


def test_projection_exposes_compact_readable_record_and_digest() -> None:
    result = select_chaser_input_provenance_proxy(_source())
    record = result.acquisition_projection_record

    assert record["schema_id"] == PROJECTION_RECORD_SCHEMA_ID
    assert record["schema_version"] == PROJECTION_RECORD_SCHEMA_VERSION
    assert record["recording_id"] == "recording-1"
    assert record["policy_id"] == PROXY_POLICY_ID
    assert record["temporal_alignment_requirement"] == (
        TEMPORAL_ALIGNMENT_REQUIREMENT
    )
    assert record["temporal_alignment_class"] == TEMPORAL_ALIGNMENT_CLASS
    assert record["physical_presentation_verified"] is False
    assert record["presentation_timestamp_available"] is False
    assert record["camera_presentation_clock_transform_available"] is False
    assert record["camera_exposure_reference"] == "unknown"
    assert record["scientific_use_class"] == "exploratory_proxy"
    assert record["behavioral_denominator"] == BEHAVIORAL_DENOMINATOR
    assert record["native_sample_count"] == 5
    assert record["unique_acquisition_frame_count"] == 3
    assert record["selected_acquisition_frame_count"] == 3
    assert record["chaser_count"] == 2
    assert record["source_manifest_sha256"] == "a" * 64
    assert record["source_verification_digest"] == "b" * 64
    assert result.acquisition_projection_digest == result.acquisition_projection_record_sha256
    assert len(result.acquisition_projection_record_sha256) == 64
    assert result.provenance["acquisition_projection_record_sha256"] == result.acquisition_projection_digest
    # The record is compact metadata; no row arrays or coordinates are embedded.
    assert "selected_chaser_position_xy" not in record
    assert "candidate_native_sample_row_index" not in record
