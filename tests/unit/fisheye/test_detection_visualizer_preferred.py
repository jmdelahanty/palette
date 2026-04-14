from __future__ import annotations

from typing import Any

import numpy as np

from fisheye.visualization import detection_visualizer as mod


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


def test_build_frame_to_detection_map_groups_indices_by_frame() -> None:
    mapping = mod._build_frame_to_detection_map(np.asarray([5, 3, 5, 9], dtype=np.int64))

    assert mapping == {5: [0, 2], 3: [1], 9: [3]}


def test_load_curated_detection_source_returns_bound_runs_and_frame_map() -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_manual"
    refined = refined_parent.create_group("refined_detect_manual")
    refined.create_array(
        "refined_row_ids",
        data=np.asarray([0, 1], dtype=np.int64),
        overwrite=True,
    )
    refined.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]], dtype=np.float64),
        overwrite=True,
    )
    refined.create_array(
        "frame_indices",
        data=np.asarray([10, 12], dtype=np.int64),
        overwrite=True,
    )
    refined.create_array("entity_ids", data=np.asarray([0, 0], dtype=np.int32), overwrite=True)
    refined.create_array(
        "bbox_img_xyxy",
        data=np.asarray([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]], dtype=np.float64),
        overwrite=True,
    )
    for name in (
        "status_codes",
        "source_kind_codes",
        "source_sparse_row_index",
        "review_state_codes",
        "keypoints_state_codes",
        "subject_mask_state_codes",
        "eye_mask_state_codes",
        "swim_bladder_state_codes",
    ):
        refined.create_array(name, data=np.asarray([0, 0], dtype=np.int8), overwrite=True)
    refined.create_array(
        "source_sparse_group_codes",
        data=np.asarray([1, 3], dtype=np.int8),
        overwrite=True,
    )
    refined.create_array(
        "manual_edit_flags",
        data=np.asarray([False, True], dtype=bool),
        overwrite=True,
    )
    source_detections = refined.create_group("source_detections")
    source_detections.create_array(
        "source_detect_row_index",
        data=np.asarray([0, 1, 2], dtype=np.int32),
        overwrite=True,
    )
    source_detections.create_array(
        "frame_indices",
        data=np.asarray([10, 12, 12], dtype=np.int32),
        overwrite=True,
    )
    source_detections.create_array(
        "bbox_img_xyxy",
        data=np.asarray(
            [[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0], [31.0, 31.0, 39.0, 39.0]],
            dtype=np.float64,
        ),
        overwrite=True,
    )
    source_detections.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1], [0.26, 0.26, 0.1, 0.1]],
            dtype=np.float64,
        ),
        overwrite=True,
    )
    source_detections.create_array(
        "decision_codes",
        data=np.asarray([0, 3, 1], dtype=np.int8),
        overwrite=True,
    )
    source_detections.create_array(
        "resolved_refined_row_id",
        data=np.asarray([0, -1, -1], dtype=np.int64),
        overwrite=True,
    )

    payload = mod._load_curated_detection_source(root)  # type: ignore[arg-type]

    assert payload is not None
    assert payload["curated_detect_run"] == "refined_detect_manual"
    assert payload["curated_crop_run"] is None
    assert payload["bbox_coords"].shape == (2, 4)
    assert payload["frame_indices"].tolist() == [10, 12]
    assert payload["frame_map"] == {10: [0], 12: [1]}
    assert payload["manual_edited_count"] == 1
    assert payload["source_detection_summary"] == {
        "total_candidates": 3,
        "decision_accepted": 1,
        "decision_filtered": 1,
        "decision_duplicate": 0,
        "decision_manual_clear": 1,
    }


def test_load_curated_detection_source_returns_none_when_group_absent() -> None:
    root = _FakeGroup()

    assert mod._load_curated_detection_source(root) is None  # type: ignore[arg-type]


def test_refined_variant_helpers_mark_canonical_and_legacy_variants() -> None:
    assert mod._describe_refined_variant("refined") == "canonical curated surface"
    assert mod._describe_refined_variant("filtered") == "legacy filtered subgroup"
    assert mod._describe_refined_variant("manual_review_001", is_manual=True) == "legacy manual subgroup"
    assert (
        mod._format_refined_stage_label("refined_detect_001", "refined")
        == "refined_detect_001 (canonical curated surface)"
    )
    assert (
        mod._format_refined_stage_label("refined_detect_001", "manual_review_001", is_manual=True)
        == "refined_detect_001/manual_review_001 (legacy manual subgroup)"
    )
