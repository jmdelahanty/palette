from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fisheye.shared.row_lineage import (
    ROW_LINEAGE_ARRAYS,
    assert_row_lineage_alignment_equal,
    assert_row_lineage_sources_equal,
    copy_row_lineage_arrays,
    copy_row_lineage_arrays_from_sources,
    copy_row_lineage_arrays_with_fallback,
)


class _FakeArray:
    def __init__(self, data: Any, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.chunks = chunks

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self) -> None:
        self._children: dict[str, _FakeArray] = {}
        self.attrs: dict[str, Any] = {}

    def create_array(
        self,
        name: str,
        *,
        data: Any,
        chunks: tuple[int, ...] | None = None,
        overwrite: bool = False,
    ) -> _FakeArray:
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        array = _FakeArray(data, chunks=chunks)
        self._children[name] = array
        return array

    def get(self, name: str):
        return self._children.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._children

    def __delitem__(self, name: str) -> None:
        del self._children[name]

    def __getitem__(self, name: str) -> _FakeArray:
        return self._children[name]


def _seed_lineage(group: _FakeGroup, *, detection_indices: np.ndarray | None = None) -> None:
    group.create_array("frame_indices", data=np.array([0, 1, 1], dtype=np.int32), chunks=(2,), overwrite=True)
    group.create_array("source_frame_indices", data=np.array([0, 5000, 5000], dtype=np.int64), chunks=(2,), overwrite=True)
    group.create_array("source_clip_indices", data=np.array([0, 1, 1], dtype=np.int32), chunks=(2,), overwrite=True)
    group.create_array(
        "source_clip_local_frame_indices",
        data=np.array([0, 12, 12], dtype=np.int64),
        chunks=(2,),
        overwrite=True,
    )
    group.create_array("frame_counts", data=np.array([1, 2], dtype=np.int32), chunks=(2,), overwrite=True)
    group.create_array(
        "detection_indices",
        data=np.array([10, 11, 12], dtype=np.int32) if detection_indices is None else detection_indices,
        chunks=(2,),
        overwrite=True,
    )
    group.create_array("instance_key", data=np.array([1001, 1002, 1003], dtype=np.uint64), chunks=(2,), overwrite=True)
    group.create_array("source_crop_row_ids", data=np.array([0, 1, 2], dtype=np.int64), chunks=(2,), overwrite=True)
    group.create_array("source_refined_row_ids", data=np.array([100, 101, 102], dtype=np.int64), chunks=(2,), overwrite=True)
    group.create_array("source_detect_row_index", data=np.array([4, 5, -1], dtype=np.int32), chunks=(2,), overwrite=True)


def test_copy_row_lineage_arrays_copies_canonical_identity() -> None:
    source = _FakeGroup()
    target = _FakeGroup()
    _seed_lineage(source)

    result = copy_row_lineage_arrays(target, source, total_rois=3)

    assert result.copied == ROW_LINEAGE_ARRAYS
    assert result.missing == ()
    assert target["source_frame_indices"][:].tolist() == [0, 5000, 5000]
    assert target["source_clip_indices"][:].tolist() == [0, 1, 1]
    assert target["source_clip_local_frame_indices"][:].tolist() == [0, 12, 12]
    assert target["instance_key"][:].tolist() == [1001, 1002, 1003]
    assert target["source_crop_row_ids"][:].tolist() == [0, 1, 2]
    assert target["source_refined_row_ids"][:].tolist() == [100, 101, 102]
    assert target["source_detect_row_index"][:].tolist() == [4, 5, -1]


def test_copy_row_lineage_arrays_rejects_bad_frame_counts() -> None:
    source = _FakeGroup()
    target = _FakeGroup()
    _seed_lineage(source)
    source.create_array("frame_counts", data=np.array([1, 1], dtype=np.int32), overwrite=True)

    with pytest.raises(ValueError, match="frame_counts.*sums to 2, expected 3"):
        copy_row_lineage_arrays(target, source, total_rois=3)


def test_copy_row_lineage_arrays_with_fallback_uses_matching_fallback() -> None:
    crop = _FakeGroup()
    keypoints = _FakeGroup()
    target = _FakeGroup()
    crop.create_array("frame_indices", data=np.array([0, 1, 1], dtype=np.int32), overwrite=True)
    crop.create_array("source_frame_indices", data=np.array([0, 5000, 5000], dtype=np.int64), overwrite=True)
    crop.create_array("source_clip_indices", data=np.array([0, 1, 1], dtype=np.int32), overwrite=True)
    crop.create_array("source_clip_local_frame_indices", data=np.array([0, 12, 12], dtype=np.int64), overwrite=True)
    crop.create_array("frame_counts", data=np.array([1, 2], dtype=np.int32), overwrite=True)
    keypoints.create_array("detection_indices", data=np.array([10, 11, 12], dtype=np.int32), overwrite=True)
    keypoints.create_array("source_crop_row_ids", data=np.array([0, 1, 2], dtype=np.int64), overwrite=True)
    keypoints.create_array("source_refined_row_ids", data=np.array([100, 101, 102], dtype=np.int64), overwrite=True)

    result = copy_row_lineage_arrays_with_fallback(target, crop, keypoints, total_rois=3)

    assert result.fallback_copied == ("detection_indices", "source_crop_row_ids", "source_refined_row_ids")
    assert target["source_crop_row_ids"][:].tolist() == [0, 1, 2]
    assert target["source_refined_row_ids"][:].tolist() == [100, 101, 102]


def test_copy_row_lineage_arrays_from_sources_accepts_resolved_arrays() -> None:
    source = _FakeGroup()
    target = _FakeGroup()
    _seed_lineage(source)

    result = copy_row_lineage_arrays_from_sources(
        target,
        {
            "frame_indices": source["frame_indices"],
            "source_frame_indices": source["source_frame_indices"],
            "source_clip_indices": source["source_clip_indices"],
            "source_clip_local_frame_indices": source["source_clip_local_frame_indices"],
            "frame_counts": source["frame_counts"],
            "detection_indices": source["detection_indices"],
            "instance_key": source["instance_key"],
            "source_crop_row_ids": source["source_crop_row_ids"],
            "source_refined_row_ids": source["source_refined_row_ids"],
            "source_detect_row_index": source["source_detect_row_index"],
        },
        total_rois=3,
    )

    assert result.copied == ROW_LINEAGE_ARRAYS
    assert target["source_refined_row_ids"][:].tolist() == [100, 101, 102]


def test_copy_row_lineage_arrays_with_fallback_rejects_mismatch() -> None:
    crop = _FakeGroup()
    keypoints = _FakeGroup()
    target = _FakeGroup()
    _seed_lineage(crop)
    _seed_lineage(keypoints, detection_indices=np.array([10, 99, 12], dtype=np.int32))

    with pytest.raises(ValueError, match="row-lineage mismatch.*detection_indices"):
        copy_row_lineage_arrays_with_fallback(target, crop, keypoints, total_rois=3)


def test_assert_row_lineage_alignment_allows_missing_optional_identity() -> None:
    reference = _FakeGroup()
    other = _FakeGroup()
    _seed_lineage(reference)
    other.create_array("frame_indices", data=reference["frame_indices"][:], overwrite=True)
    other.create_array("frame_counts", data=reference["frame_counts"][:], overwrite=True)
    other.create_array("detection_indices", data=reference["detection_indices"][:], overwrite=True)

    assert_row_lineage_alignment_equal(reference, other)


def test_assert_row_lineage_alignment_rejects_optional_identity_mismatch_when_present() -> None:
    reference = _FakeGroup()
    other = _FakeGroup()
    _seed_lineage(reference)
    _seed_lineage(other)
    other.create_array("source_refined_row_ids", data=np.array([100, 999, 102], dtype=np.int64), overwrite=True)

    with pytest.raises(ValueError, match="Alignment mismatch for source_refined_row_ids"):
        assert_row_lineage_alignment_equal(reference, other)


def test_assert_row_lineage_sources_equal_uses_resolved_arrays() -> None:
    reference = _FakeGroup()
    other = _FakeGroup()
    _seed_lineage(reference)
    _seed_lineage(other)

    assert_row_lineage_sources_equal(
        {
            "frame_indices": reference["frame_indices"],
            "source_frame_indices": reference["source_frame_indices"],
            "source_clip_indices": reference["source_clip_indices"],
            "source_clip_local_frame_indices": reference["source_clip_local_frame_indices"],
            "frame_counts": reference["frame_counts"],
            "detection_indices": reference["detection_indices"],
            "instance_key": reference["instance_key"],
            "source_crop_row_ids": reference["source_crop_row_ids"],
            "source_refined_row_ids": reference["source_refined_row_ids"],
            "source_detect_row_index": reference["source_detect_row_index"],
        },
        {
            "frame_indices": other["frame_indices"],
            "source_frame_indices": other["source_frame_indices"],
            "source_clip_indices": other["source_clip_indices"],
            "source_clip_local_frame_indices": other["source_clip_local_frame_indices"],
            "frame_counts": other["frame_counts"],
            "detection_indices": other["detection_indices"],
            "instance_key": other["instance_key"],
            "source_crop_row_ids": other["source_crop_row_ids"],
            "source_refined_row_ids": other["source_refined_row_ids"],
            "source_detect_row_index": other["source_detect_row_index"],
        },
    )


def test_assert_row_lineage_sources_equal_allows_reordered_rows_with_matching_instance_keys() -> None:
    reference = _FakeGroup()
    other = _FakeGroup()
    _seed_lineage(reference)
    order = np.array([2, 0, 1], dtype=np.int64)
    for name in ROW_LINEAGE_ARRAYS:
        data = reference[name][:]
        if name == "frame_counts":
            other.create_array(name, data=data, overwrite=True)
        else:
            other.create_array(name, data=data[order], overwrite=True)

    assert_row_lineage_sources_equal(
        {name: reference[name] for name in ROW_LINEAGE_ARRAYS},
        {name: other[name] for name in ROW_LINEAGE_ARRAYS},
    )


def test_assert_row_lineage_sources_equal_rejects_instance_key_mismatch() -> None:
    reference = _FakeGroup()
    other = _FakeGroup()
    _seed_lineage(reference)
    _seed_lineage(other)
    other.create_array("instance_key", data=np.array([1001, 9999, 1003], dtype=np.uint64), overwrite=True)

    with pytest.raises(ValueError, match="Alignment mismatch for instance_key"):
        assert_row_lineage_sources_equal(
            {name: reference[name] for name in ROW_LINEAGE_ARRAYS},
            {name: other[name] for name in ROW_LINEAGE_ARRAYS},
        )


def test_assert_row_lineage_sources_equal_legacy_without_instance_key_uses_positional_fallback() -> None:
    reference = _FakeGroup()
    other = _FakeGroup()
    _seed_lineage(reference)
    _seed_lineage(other)
    del reference["instance_key"]
    del other["instance_key"]

    assert_row_lineage_sources_equal(
        {name: reference.get(name) for name in ROW_LINEAGE_ARRAYS},
        {name: other.get(name) for name in ROW_LINEAGE_ARRAYS},
    )
