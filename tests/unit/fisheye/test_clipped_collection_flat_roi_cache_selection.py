import pytest

from fisheye.shared.clipped_collection_flat_roi_cache import (
    _build_selection,
    _collection_summary_for_selection,
    _filter_selected_runs,
)


def _selected_runs():
    return [
        {"clip_id": "clip_000000", "work_unit_id": "wu0", "camera_serial": "2010095"},
        {"clip_id": "clip_000001", "work_unit_id": "wu1", "camera_serial": "2010095"},
        {"clip_id": "clip_000002", "work_unit_id": "wu2", "camera_serial": "2010095"},
    ]


def test_filter_selected_runs_by_clip_id() -> None:
    filtered = _filter_selected_runs(_selected_runs(), clip_ids=["clip_000001"])

    assert [row["clip_id"] for row in filtered] == ["clip_000001"]
    assert [row["work_unit_id"] for row in filtered] == ["wu1"]


def test_filter_selected_runs_by_work_unit_id() -> None:
    filtered = _filter_selected_runs(_selected_runs(), work_unit_ids=["wu2"])

    assert [row["clip_id"] for row in filtered] == ["clip_000002"]
    assert [row["work_unit_id"] for row in filtered] == ["wu2"]


def test_filter_selected_runs_requires_all_requested_dimensions_to_match() -> None:
    filtered = _filter_selected_runs(
        _selected_runs(),
        clip_ids=["clip_000001"],
        work_unit_ids=["wu1"],
    )

    assert [row["clip_id"] for row in filtered] == ["clip_000001"]


def test_filter_selected_runs_fails_loud_when_filter_matches_nothing() -> None:
    with pytest.raises(ValueError, match="matched no runs"):
        _filter_selected_runs(_selected_runs(), clip_ids=["missing_clip"])


def test_collection_summary_for_selection_records_filtered_subset() -> None:
    selection = _build_selection(clip_ids=["clip_000000", "clip_000002"], work_unit_ids=None)

    summary = _collection_summary_for_selection({"selected_runs": _selected_runs()}, selection)

    assert summary["selected_run_count"] == 2
    assert [row["clip_id"] for row in summary["selected_runs"]] == ["clip_000000", "clip_000002"]
    assert summary["selection"] == {
        "clip_ids": ["clip_000000", "clip_000002"],
        "work_unit_ids": [],
        "filtered": True,
    }
