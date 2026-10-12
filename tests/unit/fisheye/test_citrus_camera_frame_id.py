from __future__ import annotations

import numpy as np

from fisheye.analysis.chaser_state_interpolator import analyze_frame_gaps, interpolate_metadata
from fisheye.shared.citrus_camera_frame_id import parent_frame_index_from_citrus_camera_frame_id


def test_citrus_camera_frame_id_is_one_based_recording_frame_id() -> None:
    assert parent_frame_index_from_citrus_camera_frame_id(1) == 0
    assert parent_frame_index_from_citrus_camera_frame_id(np.uint64(860)) == 859
    # 0 is written before Citrus has received any camera frame id.
    assert parent_frame_index_from_citrus_camera_frame_id(0) is None
    assert parent_frame_index_from_citrus_camera_frame_id(-1) is None


def _frame_metadata(camera_ids: list[int]) -> np.ndarray:
    rows = np.zeros(
        len(camera_ids),
        dtype=[
            ("stimulus_frame_num", "<u8"),
            ("triggering_camera_frame_id", "<u8"),
            ("timestamp_ns_epoch", "<i8"),
        ],
    )
    rows["stimulus_frame_num"] = np.arange(len(camera_ids))
    rows["triggering_camera_frame_id"] = camera_ids
    rows["timestamp_ns_epoch"] = 1_000_000_000 + np.arange(len(camera_ids)) * 8_333_333
    return rows


def test_unknown_id_zero_does_not_anchor_the_gap_fill() -> None:
    # Blindfish 2026-04-29 shape: stimulus rows before the first camera id, then
    # ids starting at 698. Filling from 0 invented 697 camera frames.
    metadata = _frame_metadata([0, 0, 698, 698, 699, 699, 701, 701])

    stats = analyze_frame_gaps(metadata, None)
    assert stats is not None
    assert stats.gap_ranges == [(700, 700)]

    filled, _mask = interpolate_metadata(metadata, stats, None)
    filled_ids = set(int(value) for value in filled["triggering_camera_frame_id"])
    assert filled_ids == {0, 698, 699, 700, 701}


def test_gap_analysis_skips_metadata_with_no_known_camera_id() -> None:
    assert analyze_frame_gaps(_frame_metadata([0, 0, 0]), None) is None
