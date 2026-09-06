"""Exact keyed frame correspondence, independent of Parquet physical order."""

from types import SimpleNamespace

import pyarrow as pa
import pytest

from fisheye.shared import clipped_video_collection as collection


def _units(tmp_path, monkeypatch, order=None, **changes):
    columns = {
        "camera_serial": ["2010093"] * 6,
        "clip_id": ["clip_0"] * 4 + ["clip_1"] * 2,
        "clip_index": [0] * 4 + [1] * 2,
        "video_path": ["clips/clip_0/video.mp4"] * 4 + ["clips/clip_1/video.mp4"] * 2,
        "parent_frame_index": [0, 1, 2, 3, 4, 5],
        "clip_local_frame_index": [0, 1, 2, 3, 0, 1],
    }
    columns.update(changes)
    table = pa.table(columns)
    if order is not None:
        table = table.take(order)
    monkeypatch.setattr(
        collection.pq,
        "ParquetFile",
        lambda _: SimpleNamespace(schema_arrow=table.schema),
    )
    monkeypatch.setattr(collection.pq, "read_table", lambda *a, **kw: table)
    return collection._frame_index_units(tmp_path / "index.parquet")


@pytest.mark.parametrize("order", [None, [5, 0, 3, 4, 2, 1], [5, 4, 3, 2, 1, 0]])
def test_exact_frame_pairs_are_valid_in_any_physical_row_order(
    tmp_path, monkeypatch, order
):
    assert _units(tmp_path, monkeypatch, order) == {
        ("2010093", "clip_0", 0): {
            "frame_start": 0,
            "frame_stop": 4,
            "frame_count": 4,
            "video_path": "clips/clip_0/video.mp4",
        },
        ("2010093", "clip_1", 1): {
            "frame_start": 4,
            "frame_stop": 6,
            "frame_count": 2,
            "video_path": "clips/clip_1/video.mp4",
        },
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"parent_frame_index": [0, 1, 1, 3, 4, 5]},
        {"clip_local_frame_index": [0, 1, 1, 3, 0, 1]},
        {"clip_local_frame_index": [0, 2, 1, 3, 0, 1]},
        {
            "parent_frame_index": [0, 1, 1, 3, 4, 5],
            "clip_local_frame_index": [0, 1, 1, 3, 0, 1],
        },
        {"parent_frame_index": [0, 1, 2, 3, 5, 6]},
        {"parent_frame_index": [0, 1, 2, 3, 3, 4]},
    ],
)
def test_frame_index_rejects_duplicates_gaps_overlaps_and_wrong_pairing(
    tmp_path, monkeypatch, changes
):
    with pytest.raises(ValueError, match="recording_frame_index"):
        _units(tmp_path, monkeypatch, **changes)


@pytest.mark.parametrize(
    "field", ["clip_index", "parent_frame_index", "clip_local_frame_index"]
)
@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
        [0.9, 1.9, 2.9, 3.9, 0.9, 1.9],
        [False, True, True, True, False, True],
        ["0", "1", "2", "3", "0", "1"],
        [None, 1, 2, 3, 0, 1],
        pa.array([2**63] * 6, type=pa.uint64()),
    ],
)
def test_frame_index_rejects_non_exact_integer_columns(
    tmp_path, monkeypatch, field, values
):
    with pytest.raises(ValueError, match="recording_frame_index"):
        _units(tmp_path, monkeypatch, **{field: values})


@pytest.mark.parametrize("field", ["camera_serial", "clip_id", "video_path"])
def test_frame_index_rejects_null_mapping_keys(tmp_path, monkeypatch, field):
    with pytest.raises(ValueError, match="recording_frame_index"):
        _units(tmp_path, monkeypatch, **{field: [None] * 6})
