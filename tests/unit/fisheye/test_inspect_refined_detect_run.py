from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fisheye.utils.inspect_refined_detect_run import collect_refined_detect_report


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if isinstance(existing, _FakeGroup):
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def create_array(self, name: str, *, data: Any, overwrite: bool = False) -> _FakeArray:
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        arr = _FakeArray(data)
        self._children[name] = arr
        return arr

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def group_keys(self):
        return [key for key, value in self._children.items() if isinstance(value, _FakeGroup)]

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str):
        if "/" in key:
            current: _FakeGroup | _FakeArray = self
            for token in key.split("/"):
                if not isinstance(current, _FakeGroup):
                    raise KeyError(key)
                current = current._children[token]
            return current
        return self._children[key]


def _build_root() -> _FakeGroup:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["detect_review_status"] = {"state": "approved", "intended_use": "training"}
    refined.attrs["summary_statistics"] = {
        "rows_present": 2,
        "rows_filtered_out": 1,
        "rows_missing": 0,
        "rows_manual_edited": 1,
    }

    instances = refined.create_group("instances")
    for name, data in (
        ("refined_row_ids", np.asarray([1, 4], dtype=np.int64)),
        ("frame_indices", np.asarray([1, 4], dtype=np.int32)),
        ("frame_offsets", np.asarray([0, 1, 1, 1, 2, 2], dtype=np.int64)),
        ("bbox_img_xyxy", np.asarray([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]], dtype=np.float64)),
        ("source_kind_codes", np.asarray([1, 3], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, True], dtype=bool)),
        ("source_detect_row_index", np.asarray([0, -1], dtype=np.int32)),
        ("frame_counts", np.asarray([0, 1, 0, 0, 1], dtype=np.int32)),
        ("reason", np.asarray(["clean", "manual_add"], dtype=object)),
    ):
        instances.create_array(name, data=data, overwrite=True)

    source_detections = refined.create_group("source_detections")
    for name, data in (
        ("source_detect_row_index", np.asarray([0, 1, 2], dtype=np.int32)),
        ("frame_indices", np.asarray([1, 4, 4], dtype=np.int32)),
        (
            "bbox_img_xyxy",
            np.asarray(
                [[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0], [3.1, 3.1, 4.1, 4.1]],
                dtype=np.float64,
            ),
        ),
        (
            "bbox_norm_coords",
            np.asarray(
                [[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1], [0.26, 0.26, 0.1, 0.1]],
                dtype=np.float64,
            ),
        ),
        ("decision_codes", np.asarray([0, 3, 1], dtype=np.int8)),
        ("resolved_refined_row_id", np.asarray([1, -1, -1], dtype=np.int64)),
        ("reason", np.asarray(["clean", "manual_clear", "low_score"], dtype=object)),
    ):
        source_detections.create_array(name, data=data, overwrite=True)

    return root


def test_collect_refined_detect_report_summarizes_sparse_surfaces() -> None:
    payload = collect_refined_detect_report(
        _build_root(),  # type: ignore[arg-type]
        zarr_path=Path("/tmp/refined_detect.zarr"),
        instance_limit=10,
        source_limit=10,
    )

    assert payload["refined_run"] == "refined_detect_001"
    assert payload["source_detect_run"] == "detect_001"
    assert payload["instances"]["total_instances"] == 2
    assert payload["instances"]["manual_edited_instances"] == 1
    assert payload["instances"]["source_kind_counts"] == {"manual": 1, "raw_detect": 1}
    assert payload["source_detections"]["summary"] == {
        "total_candidates": 3,
        "decision_accepted": 1,
        "decision_filtered": 1,
        "decision_duplicate": 0,
        "decision_manual_clear": 1,
    }
    assert len(payload["source_detections"]["preview"]) == 3


def test_collect_refined_detect_report_filters_source_preview_rows() -> None:
    payload = collect_refined_detect_report(
        _build_root(),  # type: ignore[arg-type]
        source_limit=10,
        source_decisions=["filtered"],
    )

    preview = payload["source_detections"]["preview"]
    assert payload["source_detections"]["summary"]["total_candidates"] == 3
    assert payload["source_detections"]["preview_decision_filter"] == ["filtered"]
    assert len(preview) == 1
    assert preview[0]["decision"] == "filtered"
    assert preview[0]["reason"] == "low_score"
