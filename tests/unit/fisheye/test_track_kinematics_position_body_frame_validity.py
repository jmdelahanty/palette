from __future__ import annotations

import numpy as np
import pytest

from fisheye.analysis import track_kinematics as mod


def _independent_validity(
    *,
    positions: np.ndarray,
    headings: np.ndarray,
    position_valid: np.ndarray,
    heading_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    return mod._build_sample_validity_arrays(
        track_id=7,
        positions_px=positions,
        headings_deg=headings,
        keypoint_success=np.ones(headings.shape, dtype=bool),
        detection_source=None,
        position_valid=position_valid,
        heading_valid=heading_valid,
        validity_profile=mod.TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE,
    )


def test_independent_heading_failure_does_not_invalidate_linear_position() -> None:
    validity = _independent_validity(
        positions=np.asarray([[10.0, 20.0], [11.0, 20.0]], dtype=np.float32),
        headings=np.asarray([0.0, np.nan], dtype=np.float32),
        position_valid=np.asarray([True, True], dtype=bool),
        heading_valid=np.asarray([True, False], dtype=bool),
    )

    assert validity["linear_sample_valid"].tolist() == [True, True]
    assert validity["angular_sample_valid"].tolist() == [True, False]
    assert validity["sample_valid"].tolist() == [True, False]
    assert validity["linear_sample_reason_code"].tolist() == [0, 0]
    assert validity["angular_sample_reason_code"].tolist() == [0, 2]


def test_independent_position_failure_does_not_invalidate_heading() -> None:
    validity = _independent_validity(
        positions=np.asarray([[np.nan, np.nan], [11.0, 20.0]], dtype=np.float32),
        headings=np.asarray([15.0, 20.0], dtype=np.float32),
        position_valid=np.asarray([False, True], dtype=bool),
        heading_valid=np.asarray([True, True], dtype=bool),
    )

    assert validity["linear_sample_valid"].tolist() == [False, True]
    assert validity["angular_sample_valid"].tolist() == [True, True]
    assert validity["sample_valid"].tolist() == [False, True]
    assert validity["linear_sample_reason_code"].tolist() == [3, 0]
    assert validity["angular_sample_reason_code"].tolist() == [0, 0]


def test_independent_profile_requires_both_exact_bool_validity_arrays() -> None:
    common = {
        "track_id": 1,
        "positions_px": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "headings_deg": np.asarray([0.0], dtype=np.float32),
        "keypoint_success": np.asarray([True], dtype=bool),
        "detection_source": None,
        "validity_profile": mod.TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE,
    }
    with pytest.raises(ValueError, match="requires both"):
        mod._build_sample_validity_arrays(
            **common,
            position_valid=np.asarray([True], dtype=bool),
        )
    with pytest.raises(ValueError, match="exact row-aligned bool"):
        mod._build_sample_validity_arrays(
            **common,
            position_valid=np.asarray([1], dtype=np.uint8),
            heading_valid=np.asarray([True], dtype=bool),
        )


def test_compatibility_profile_preserves_joint_keypoint_validity() -> None:
    validity = mod._build_sample_validity_arrays(
        track_id=3,
        positions_px=np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        headings_deg=np.asarray([0.0, 10.0], dtype=np.float32),
        keypoint_success=np.asarray([True, True], dtype=bool),
        detection_source=np.asarray([0, 1], dtype=np.int8),
    )

    assert validity["sample_valid"].tolist() == [True, False]
    assert validity["linear_sample_valid"].tolist() == [True, False]
    assert validity["angular_sample_valid"].tolist() == [True, False]


def test_track_builder_uses_angular_validity_for_turning_only() -> None:
    tracks, _summaries = mod.build_track_datasets(
        track_ids=np.asarray([0, 0, 0], dtype=np.int64),
        frames=np.asarray([0, 1, 2], dtype=np.int64),
        positions_px=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        headings_deg=np.asarray([0.0, np.nan, 20.0], dtype=np.float32),
        keypoint_success=np.asarray([True, True, True], dtype=bool),
        detection_source=None,
        fps=10.0,
        smooth_seconds=0.0,
        pixel_to_mm=None,
        position_valid=np.asarray([True, True, True], dtype=bool),
        heading_valid=np.asarray([True, False, True], dtype=bool),
        validity_profile=mod.TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE,
    )

    track = tracks[0]
    assert track["linear_sample_valid"].tolist() == [True, True, True]
    assert track["angular_sample_valid"].tolist() == [True, False, True]
    assert np.isfinite(track["speed_raw_px"][1:]).all()
    assert np.isnan(track["delta_heading_degrees"][1:]).all()
    assert track["sample_validity_profile"] == (
        mod.TRACK_SAMPLE_VALIDITY_INDEPENDENT_PROFILE
    )
