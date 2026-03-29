import numpy as np

from fisheye.analysis.track_kinematics import (
    _filter_public_track_rows,
    _ordered_track_arena_ids,
)


def test_filter_public_track_rows_excludes_unassigned_by_default() -> None:
    detection_source = np.array([0, 1, 0, 1], dtype=np.int8)

    track_ids, frames, positions_px, headings_deg, keypoint_success, filtered_source = (
        _filter_public_track_rows(
            track_ids=np.array([-1, 0, 1, -1], dtype=np.int64),
            frames=np.array([10, 11, 12, 13], dtype=np.int64),
            positions_px=np.array(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                dtype=np.float64,
            ),
            headings_deg=np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
            keypoint_success=np.array([True, False, True, False], dtype=bool),
            detection_source=detection_source,
            include_unassigned=False,
        )
    )

    assert track_ids.tolist() == [0, 1]
    assert frames.tolist() == [11, 12]
    assert positions_px.tolist() == [[2.0, 2.0], [3.0, 3.0]]
    assert headings_deg.tolist() == [10.0, 20.0]
    assert keypoint_success.tolist() == [False, True]
    assert filtered_source is not None
    assert filtered_source.tolist() == [1, 0]


def test_filter_public_track_rows_keeps_unassigned_when_requested() -> None:
    track_ids, frames, positions_px, headings_deg, keypoint_success, filtered_source = (
        _filter_public_track_rows(
            track_ids=np.array([-1, 0], dtype=np.int64),
            frames=np.array([5, 6], dtype=np.int64),
            positions_px=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
            headings_deg=np.array([15.0, 30.0], dtype=np.float64),
            keypoint_success=np.array([True, True], dtype=bool),
            detection_source=None,
            include_unassigned=True,
        )
    )

    assert track_ids.tolist() == [-1, 0]
    assert frames.tolist() == [5, 6]
    assert positions_px.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert headings_deg.tolist() == [15.0, 30.0]
    assert keypoint_success.tolist() == [True, True]
    assert filtered_source is None


def test_ordered_track_arena_ids_allows_diagnostic_unassigned_track() -> None:
    result = _ordered_track_arena_ids([-1, 0, 1], {0: 5, 1: 9})

    assert result is not None
    assert result.tolist() == [-1, 5, 9]
