from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.shared.refined_detect_identity import (
    collect_refined_detect_identity_validation,
    validate_refined_detect_identity,
)


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeArray | _FakeGroup] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def create_array(self, name: str, *, data: Any, **_kwargs: Any) -> _FakeArray:
        child = _FakeArray(data)
        self._children[name] = child
        return child

    def get(self, name: str):
        return self._children.get(name)

    def __contains__(self, key: str) -> bool:
        return key in self._children

    def __getitem__(self, key: str):
        return self._children[key]


def _add_array(group: _FakeGroup, name: str, data: Any) -> None:
    group.create_array(name, data=data)


def _seed_sparse_refined_run() -> _FakeGroup:
    refined = _FakeGroup(path="refined_detect_runs/refined_detect_001")
    refined.attrs["curated_primary_surface"] = "instances"
    refined.attrs["row_identity_policy"] = "stable_sparse_refined_row_id"
    refined.attrs["refined_storage_semantics"] = "sparse_instances_v1"

    instances = refined.create_group("instances")
    for name, data in (
        ("refined_row_ids", np.asarray([10, 11, 20], dtype=np.int64)),
        ("frame_indices", np.asarray([0, 0, 2], dtype=np.int32)),
        ("frame_offsets", np.asarray([0, 2, 2, 3], dtype=np.int64)),
        ("bbox_img_xyxy", np.asarray([[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]], dtype=np.float64)),
        (
            "bbox_norm_coords",
            np.asarray(
                [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1]],
                dtype=np.float64,
            ),
        ),
        ("source_kind_codes", np.asarray([1, 3, 2], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, True, False], dtype=bool)),
        ("source_detect_row_index", np.asarray([0, 1, -1], dtype=np.int32)),
        ("frame_counts", np.asarray([2, 0, 1], dtype=np.int32)),
    ):
        _add_array(instances, name, data)

    source = refined.create_group("source_detections")
    for name, data in (
        ("source_detect_row_index", np.asarray([0, 1, 2], dtype=np.int32)),
        ("frame_indices", np.asarray([0, 0, 1], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[1, 1, 2, 2], [3, 3, 4, 4], [7, 7, 8, 8]], dtype=np.float64)),
        (
            "bbox_norm_coords",
            np.asarray(
                [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]],
                dtype=np.float64,
            ),
        ),
        ("decision_codes", np.asarray([0, 0, 1], dtype=np.int8)),
        ("resolved_refined_row_id", np.asarray([10, 11, -1], dtype=np.int64)),
    ):
        _add_array(source, name, data)
    return refined


def _issue_codes(refined: _FakeGroup) -> set[str]:
    return {issue.code for issue in validate_refined_detect_identity(refined)}


def test_validate_refined_detect_identity_accepts_coherent_sparse_run() -> None:
    refined = _seed_sparse_refined_run()

    summary = collect_refined_detect_identity_validation(refined)

    assert summary["ok"] is True
    assert summary["error_count"] == 0
    assert summary["warning_count"] == 0


def test_validate_refined_detect_identity_rejects_duplicate_refined_row_ids() -> None:
    refined = _seed_sparse_refined_run()
    refined["instances"]._children["refined_row_ids"] = _FakeArray(
        np.asarray([10, 10, 20], dtype=np.int64)
    )

    assert "duplicate_refined_row_id" in _issue_codes(refined)


def test_validate_refined_detect_identity_rejects_unsorted_instances() -> None:
    refined = _seed_sparse_refined_run()
    refined["instances"]._children["refined_row_ids"] = _FakeArray(
        np.asarray([11, 10, 20], dtype=np.int64)
    )

    assert "instances_not_frame_sorted" in _issue_codes(refined)


def test_validate_refined_detect_identity_rejects_stale_source_resolution() -> None:
    refined = _seed_sparse_refined_run()
    refined["source_detections"]._children["resolved_refined_row_id"] = _FakeArray(
        np.asarray([10, 99, -1], dtype=np.int64)
    )

    codes = _issue_codes(refined)

    assert "accepted_source_resolves_missing_instance" in codes
    assert "source_link_mismatch" in codes


def test_validate_refined_detect_identity_warns_on_legacy_manual_pointer() -> None:
    refined = _seed_sparse_refined_run()
    refined.attrs["manual_review_latest"] = "manual"

    summary = collect_refined_detect_identity_validation(refined)

    assert summary["ok"] is True
    assert summary["warning_count"] == 1
    assert summary["issues"][0]["code"] == "legacy_manual_review_pointer_present"
