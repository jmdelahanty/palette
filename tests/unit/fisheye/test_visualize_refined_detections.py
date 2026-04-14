from __future__ import annotations

from pathlib import Path

import numpy as np

from fisheye.visualization import visualize_refined_detections as mod


def test_render_refinement_pipeline_png_uses_noninteractive_save(monkeypatch) -> None:
    fake_png = b"\x89PNG\r\n\x1a\nFAKE_REFINED"
    calls = {"count": 0}

    def _fake_visualize(
        zarr_path: str,
        refined_run=None,
        save_path=None,
        frame_range=None,
        show=True,
        show_curated=False,
        curated_detect_run=None,
    ):
        calls["count"] += 1
        assert zarr_path == "/tmp/fake.zarr"
        assert refined_run == "refined_detect_1"
        assert frame_range is None
        assert show is False
        assert show_curated is False
        assert curated_detect_run is None
        assert save_path is not None
        Path(save_path).write_bytes(fake_png)

    monkeypatch.setattr(mod, "visualize_refinement_pipeline", _fake_visualize)

    png_bytes, meta = mod.render_refinement_pipeline_png(
        "/tmp/fake.zarr",
        refined_run="refined_detect_1",
    )
    assert calls["count"] == 1
    assert png_bytes == fake_png
    assert meta["refined_run"] == "refined_detect_1"


class _FakeArray:
    def __init__(self, data) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children = {}
        self.attrs = {}
        self.path = path

    def create_group(self, name: str):
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str):
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if existing is not None:
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def create_array(self, name: str, *, data, overwrite: bool = False):
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
        return list(self._children.keys())

    def keys(self):
        return self._children.keys()

    def __contains__(self, key: str) -> bool:
        try:
            _ = self[key]
            return True
        except Exception:
            return False

    def __getitem__(self, key: str):
        if "/" in key:
            current = self
            for token in key.split("/"):
                current = current._children[token]
            return current
        return self._children[key]


def test_load_refined_stage_uses_total_frames_when_frame_counts_missing() -> None:
    group = _FakeGroup()
    group.create_array("bbox_norm_coords", data=np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float64), overwrite=True)
    group.create_array("frame_indices", data=np.asarray([9], dtype=np.int32), overwrite=True)

    payload = mod.load_refined_stage(
        group,
        "curated",
        fps=10.0,
        total_frames=20,
    )

    assert payload["total_frames"] == 20
    assert payload["frame_counts"].shape[0] == 20
    assert int(payload["frame_counts"][9]) == 1


def test_load_refined_stage_counts_manual_edited_curated_rows() -> None:
    group = _FakeGroup()
    for name, data in (
        ("refined_row_ids", np.asarray([0, 1], dtype=np.int64)),
        ("frame_indices", np.asarray([1, 4], dtype=np.int32)),
        ("entity_ids", np.asarray([0, 0], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]], dtype=np.float64)),
        ("status_codes", np.asarray([0, 0], dtype=np.int8)),
        ("source_kind_codes", np.asarray([1, 3], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, True], dtype=bool)),
        ("review_state_codes", np.asarray([1, 1], dtype=np.int8)),
        ("keypoints_state_codes", np.asarray([0, 0], dtype=np.int8)),
        ("subject_mask_state_codes", np.asarray([0, 0], dtype=np.int8)),
        ("eye_mask_state_codes", np.asarray([0, 0], dtype=np.int8)),
        ("swim_bladder_state_codes", np.asarray([0, 0], dtype=np.int8)),
        ("detection_source", np.asarray([0, 0], dtype=np.int8)),
    ):
        group.create_array(name, data=data, overwrite=True)

    payload = mod.load_refined_stage(group, "refined", fps=10.0, total_frames=6)

    assert payload["manual_edited_detections"] == 1
    assert payload["manual_edit_flags"].tolist() == [False, True]


def test_load_refined_stage_reads_sparse_instances_surface() -> None:
    group = _FakeGroup()
    instances = group.create_group("instances")
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
    ):
        instances.create_array(name, data=data, overwrite=True)
    source_detections = group.create_group("source_detections")
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
    ):
        source_detections.create_array(name, data=data, overwrite=True)

    payload = mod.load_refined_stage(group, "refined", fps=10.0, total_frames=5)

    assert payload["manual_edited_detections"] == 1
    assert payload["frame_indices"].tolist() == [1, 4]
    assert payload["frame_counts"].tolist() == [0, 1, 0, 0, 1]
    assert payload["source_detection_summary"] == {
        "total_candidates": 3,
        "decision_accepted": 1,
        "decision_filtered": 1,
        "decision_duplicate": 0,
        "decision_manual_clear": 1,
    }


def test_resolve_curated_stage_returns_curated_refined_run() -> None:
    root = _FakeGroup()
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    refined.attrs["source_detect_run"] = "detect_001"
    refined.attrs["active_sparse_group"] = "manual_a"
    for name, data in (
        ("refined_row_ids", np.asarray([0], dtype=np.int64)),
        ("frame_indices", np.asarray([1], dtype=np.int32)),
        ("entity_ids", np.asarray([0], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[10.0, 10.0, 20.0, 20.0]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64)),
        ("status_codes", np.asarray([0], dtype=np.int8)),
        ("source_kind_codes", np.asarray([2], dtype=np.int8)),
        ("source_sparse_row_index", np.asarray([0], dtype=np.int32)),
        ("source_sparse_group_codes", np.asarray([3], dtype=np.int8)),
        ("review_state_codes", np.asarray([1], dtype=np.int8)),
        ("keypoints_state_codes", np.asarray([0], dtype=np.int8)),
        ("subject_mask_state_codes", np.asarray([0], dtype=np.int8)),
        ("eye_mask_state_codes", np.asarray([0], dtype=np.int8)),
        ("swim_bladder_state_codes", np.asarray([0], dtype=np.int8)),
    ):
        refined.create_array(name, data=data, overwrite=True)

    detect_run, crop_run, detect_group = mod._resolve_curated_stage(root)  # type: ignore[arg-type]

    assert detect_run == "refined_detect_001"
    assert crop_run is None
    assert detect_group is refined


def test_iter_refined_stage_names_prefers_canonical_then_manual_then_legacy() -> None:
    refined = _FakeGroup()
    refined.attrs["manual_review_latest"] = "manual_review_001"
    for name, data in (
        ("refined_row_ids", np.asarray([0], dtype=np.int64)),
        ("frame_indices", np.asarray([1], dtype=np.int32)),
        ("entity_ids", np.asarray([0], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[10.0, 10.0, 20.0, 20.0]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.5, 0.5, 0.1, 0.1]], dtype=np.float64)),
        ("status_codes", np.asarray([0], dtype=np.int8)),
        ("source_kind_codes", np.asarray([1], dtype=np.int8)),
        ("source_sparse_row_index", np.asarray([0], dtype=np.int32)),
        ("source_sparse_group_codes", np.asarray([0], dtype=np.int8)),
        ("review_state_codes", np.asarray([1], dtype=np.int8)),
        ("keypoints_state_codes", np.asarray([0], dtype=np.int8)),
        ("subject_mask_state_codes", np.asarray([0], dtype=np.int8)),
        ("eye_mask_state_codes", np.asarray([0], dtype=np.int8)),
        ("swim_bladder_state_codes", np.asarray([0], dtype=np.int8)),
    ):
        refined.create_array(name, data=data, overwrite=True)

    manual = refined.create_group("manual_review_001")
    manual.create_array("bbox_norm_coords", data=np.asarray([[0.1, 0.1, 0.1, 0.1]], dtype=np.float64), overwrite=True)
    manual.create_array("frame_indices", data=np.asarray([1], dtype=np.int32), overwrite=True)
    filtered = refined.create_group("filtered")
    filtered.create_array("bbox_norm_coords", data=np.asarray([[0.1, 0.1, 0.1, 0.1]], dtype=np.float64), overwrite=True)
    filtered.create_array("frame_indices", data=np.asarray([1], dtype=np.int32), overwrite=True)
    interpolated = refined.create_group("interpolated")
    interpolated.create_array("bbox_norm_coords", data=np.asarray([[0.1, 0.1, 0.1, 0.1]], dtype=np.float64), overwrite=True)
    interpolated.create_array("frame_indices", data=np.asarray([1], dtype=np.int32), overwrite=True)

    assert mod._iter_refined_stage_names(refined) == [
        "refined",
        "manual_review_001",
        "filtered",
        "interpolated",
    ]


def test_stage_display_name_marks_legacy_variants_explicitly() -> None:
    assert mod._stage_display_name("refined", None) == "Canonical Curated Refined Surface"
    assert mod._stage_display_name("filtered", None) == "Legacy Filtered Subgroup"
    assert mod._stage_display_name("interpolated", None) == "Legacy Interpolated Subgroup"
    assert mod._stage_display_name("manual", "manual") == "Legacy Manual Subgroup"
