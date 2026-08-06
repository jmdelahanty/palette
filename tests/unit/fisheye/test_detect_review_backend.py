from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import zarr

from fisheye.shared.zarr_run_completion import AUTHORITATIVE_RUN_ATTR, mark_run_complete
from fisheye.tune import detect_review_backend as backend


class _FakeGroup:
    def __init__(self, *, path: str = "") -> None:
        self._children: dict[str, _FakeGroup | _FakeArray] = {}
        self.attrs: dict[str, Any] = {}
        self.path = path

    def create_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).create_group(tail)
        if name in self._children:
            raise ValueError(f"{name} already exists")
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def require_group(self, name: str) -> "_FakeGroup":
        if "/" in name:
            head, tail = name.split("/", 1)
            return self.require_group(head).require_group(tail)
        existing = self._children.get(name)
        if existing is not None:
            if not isinstance(existing, _FakeGroup):
                raise ValueError(f"{name} exists and is not a group")
            return existing
        child_path = f"{self.path}/{name}" if self.path else name
        child = _FakeGroup(path=child_path)
        self._children[name] = child
        return child

    def get(self, name: str):
        try:
            return self[name]
        except Exception:
            return None

    def keys(self):
        return self._children.keys()

    def create_array(
        self,
        name: str,
        *,
        data: Any | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        fill_value: Any = 0,
        overwrite: bool = False,
        **_kwargs: Any,
    ):
        if not overwrite and name in self._children:
            raise ValueError(f"{name} already exists")
        if data is None:
            if shape is None:
                raise ValueError("shape is required when data is omitted")
            try:
                resolved_dtype = np.dtype(dtype) if dtype is not None else np.float32
            except TypeError:
                resolved_dtype = object
            data = np.full(shape, fill_value, dtype=resolved_dtype)
        arr = _FakeArray(data)
        self._children[name] = arr
        return arr

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

    def __delitem__(self, key: str) -> None:
        if "/" in key:
            head, tail = key.split("/", 1)
            child = self._children[head]
            if not isinstance(child, _FakeGroup):
                raise KeyError(key)
            del child[tail]
            return
        del self._children[key]


class _FakeArray:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value) -> None:
        self._data[item] = value


def _seed_dense_curated_run(run: _FakeGroup) -> None:
    for name, data in (
        ("refined_row_ids", np.asarray([0, 1, 2], dtype=np.int64)),
        ("frame_indices", np.asarray([0, 1, 2], dtype=np.int32)),
        ("entity_ids", np.asarray([0, 0, 0], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[1, 1, 2, 2], [3, 3, 4, 4], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float64)),
        ("status_codes", np.asarray([0, 0, 2], dtype=np.int8)),
        ("source_kind_codes", np.asarray([1, 3, 0], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, True, True], dtype=bool)),
        ("source_detect_row_index", np.asarray([10, -1, 12], dtype=np.int32)),
        ("review_state_codes", np.asarray([1, 1, 1], dtype=np.int8)),
        ("keypoints_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("subject_mask_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("eye_mask_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("swim_bladder_state_codes", np.asarray([0, 0, 0], dtype=np.int8)),
        ("confidence_scores", np.asarray([0.9, 0.8, np.nan], dtype=np.float32)),
        ("class_ids", np.asarray([0, 0, -1], dtype=np.int32)),
        ("detection_source", np.asarray([0, 0, 0], dtype=np.int8)),
        ("reason", np.asarray(["clean", "manual_correction", "manual_clear"], dtype=object)),
    ):
        run.create_array(name, data=data, overwrite=True)
    source = run.create_group("source_detections")
    for name, data in (
        ("source_detect_row_index", np.asarray([10, 11, 12], dtype=np.int32)),
        ("frame_indices", np.asarray([0, 1, 2], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.1, 0.1], [0.3, 0.3, 0.1, 0.1]], dtype=np.float64)),
        ("decision_codes", np.asarray([0, 0, 3], dtype=np.int8)),
        ("resolved_refined_row_id", np.asarray([0, 1, -1], dtype=np.int64)),
        ("confidence_scores", np.asarray([0.9, 0.8, 0.4], dtype=np.float32)),
        ("class_ids", np.asarray([0, 0, 0], dtype=np.int32)),
        ("review_notes", np.asarray(["", "", ""], dtype=object)),
        ("reason", np.asarray(["clean", "clean", "manual_clear"], dtype=object)),
    ):
        source.create_array(name, data=data, overwrite=True)
    run.attrs["curated_row_storage"] = "dense_frame_entity_v3"


def _make_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs["width"] = 5
    root.attrs["height"] = 4
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5))
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_001"
    refined = refined_parent.create_group("refined_detect_001")
    _seed_dense_curated_run(refined)
    return root


def _make_multi_instance_root() -> _FakeGroup:
    root = _FakeGroup()
    root.attrs.update({"width": 5, "height": 4, "total_frames": 3})
    raw = root.create_group("raw_video")
    raw.create_array("images_ds", data=np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5))
    refined_parent = root.create_group("refined_detect_runs")
    refined_parent.attrs["latest"] = "refined_detect_multi"
    refined = refined_parent.create_group("refined_detect_multi")
    instances = refined.create_group("instances")
    for name, data in (
        ("refined_row_ids", np.asarray([10, 11, 20], dtype=np.int64)),
        ("frame_indices", np.asarray([0, 0, 2], dtype=np.int32)),
        ("frame_offsets", np.asarray([0, 2, 2, 3], dtype=np.int64)),
        ("bbox_img_xyxy", np.asarray([[0, 0, 1, 1], [2, 0, 3, 1], [1, 2, 2, 3]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.2, 0.2, 0.2, 0.2], [0.6, 0.2, 0.2, 0.2], [0.4, 0.7, 0.2, 0.2]], dtype=np.float64)),
        ("source_kind_codes", np.asarray([1, 1, 3], dtype=np.int8)),
        ("manual_edit_flags", np.asarray([False, False, True], dtype=bool)),
        ("source_detect_row_index", np.asarray([0, 1, -1], dtype=np.int32)),
        ("frame_counts", np.asarray([2, 0, 1], dtype=np.int32)),
        ("confidence_scores", np.asarray([0.9, 0.8, 1.0], dtype=np.float32)),
        ("class_ids", np.asarray([0, 0, 0], dtype=np.int32)),
        ("instance_key", np.asarray([101, 102, 201], dtype=np.uint64)),
        ("instance_key_origin_codes", np.asarray([1, 1, 2], dtype=np.int8)),
        ("reason", np.asarray(["clean", "clean", "manual_correction"], dtype=object)),
    ):
        instances.create_array(name, data=data, overwrite=True)
    source = refined.create_group("source_detections")
    for name, data in (
        ("source_detect_row_index", np.asarray([0, 1], dtype=np.int32)),
        ("frame_indices", np.asarray([0, 0], dtype=np.int32)),
        ("bbox_img_xyxy", np.asarray([[0, 0, 1, 1], [2, 0, 3, 1]], dtype=np.float64)),
        ("bbox_norm_coords", np.asarray([[0.2, 0.2, 0.2, 0.2], [0.6, 0.2, 0.2, 0.2]], dtype=np.float64)),
        ("decision_codes", np.asarray([0, 0], dtype=np.int8)),
        ("resolved_refined_row_id", np.asarray([10, 11], dtype=np.int64)),
        ("confidence_scores", np.asarray([0.9, 0.8], dtype=np.float32)),
        ("class_ids", np.asarray([0, 0], dtype=np.int32)),
        ("review_notes", np.asarray(["", ""], dtype=object)),
        ("reason", np.asarray(["clean", "clean"], dtype=object)),
    ):
        source.create_array(name, data=data, overwrite=True)
    return root


def test_resolve_context_loads_default_failure_frame(monkeypatch) -> None:
    root = _make_root()
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)

    session = backend.resolve_review_context("/tmp/example.zarr")
    payload = backend.load_frame_payload(session, 0)

    assert session.review_rows.tolist() == [2]
    assert payload["frame_idx"] == 2
    assert payload["bbox_norm"] is None
    assert payload["status"]["status_label"] == "filtered_out"
    assert payload["frame_image"]["shape"] == [4, 5]
    assert payload["frame_image"]["encoding"] == "base64_raw"


def test_resolve_context_loads_complete_multi_detection_frame(monkeypatch) -> None:
    root = _make_multi_instance_root()
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)

    session = backend.resolve_review_context("/tmp/multi.zarr", include_all=True)
    payload = backend.load_frame_payload(session, 0)
    empty_payload = backend.load_frame_payload(session, 1)
    summary = backend.review_session_summary(session)

    assert session.review_axis == "frame_instances"
    assert session.review_rows.tolist() == [0, 1, 2]
    assert payload["frame_idx"] == 0
    assert payload["detection_count"] == 2
    assert [item["instance_key"] for item in payload["detections"]] == ["101", "102"]
    assert empty_payload["frame_idx"] == 1
    assert empty_payload["detections"] == []
    assert empty_payload["status"]["status_label"] == "missing"
    assert empty_payload["frame_label_state"] == "unreviewed"
    assert empty_payload["frame_label_reason"] == "none"
    assert summary["total_instances"] == 3
    assert summary["multi_instance_frames"] == 1
    assert summary["max_instances_per_frame"] == 2


def test_empty_collection_does_not_implicitly_review_empty_frame(monkeypatch) -> None:
    root = _make_multi_instance_root()
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
    session = backend.resolve_review_context("/tmp/multi.zarr", include_all=True)

    with np.testing.assert_raises_regex(ValueError, "Use Mark negative"):
        backend.apply_detection_collection(session, position=1, detections=[])


def test_mark_negative_and_completion_guard(monkeypatch) -> None:
    root = _make_multi_instance_root()
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)

    def _mark(*_args, n_frames, frame_index, **_kwargs):
        decision_codes = np.zeros(n_frames, dtype=np.uint8)
        reason_codes = np.zeros(n_frames, dtype=np.uint16)
        decision_codes[frame_index] = 1
        reason_codes[frame_index] = 1
        return SimpleNamespace(decision_codes=decision_codes, reason_codes=reason_codes)

    monkeypatch.setattr(backend, "set_detect_frame_negative", _mark)
    session = backend.resolve_review_context("/tmp/multi.zarr", include_all=True)

    before = backend.detect_frame_review_completion_guard(session)
    result = backend.apply_negative_frame_decision(session, position=1)
    after = backend.detect_frame_review_completion_guard(session)

    assert before["ready"] is False
    assert before["unresolved_frames"] == [1]
    assert result == {
        "action": "mark_negative_frame",
        "frame_idx": 1,
        "detection_count": 0,
        "frame_label_state": "negative",
        "frame_label_reason": "subject_outside_dish",
    }
    assert after["ready"] is True
    assert after["negative_frame_count"] == 1


def test_apply_detection_collection_preserves_keys_adds_and_deletes(monkeypatch) -> None:
    root = _make_multi_instance_root()
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
    captured: dict[str, Any] = {}

    def _capture_write(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(backend.detect_review_mod, "_write_dense_curated_edit_payload", _capture_write)
    monkeypatch.setattr(
        backend,
        "_reload_payload",
        lambda session: setattr(session, "payload", captured["payload"]),
    )

    session = backend.resolve_review_context("/tmp/multi.zarr", include_all=True)
    result = backend.apply_detection_collection(
        session,
        position=0,
        detections=[
            {"instance_key": "101", "bbox_norm": [0.25, 0.25, 0.2, 0.2], "class_id": 0},
            {"instance_key": None, "bbox_norm": [0.8, 0.25, 0.1, 0.1], "class_id": 0},
        ],
    )

    written = captured["payload"]
    frame_zero = np.asarray(written["frame_indices"]) == 0
    assert result["action"] == "replace_detection_collection"
    assert result["added"] == 1
    assert result["updated"] == 1
    assert result["removed"] == 1
    assert np.asarray(written["refined_row_ids"])[frame_zero].tolist() == [10, -1]
    assert np.asarray(written["instance_keys"], dtype=np.uint64)[frame_zero].tolist() == [101, 0]
    assert np.asarray(written["source_surface_decision_labels"], dtype=object).tolist() == [
        "accepted",
        "manual_clear",
    ]
    assert captured["source_context"]["edit_mode"] == "frame_instance_collection"


def test_apply_detection_collection_rejects_key_from_another_frame(monkeypatch) -> None:
    root = _make_multi_instance_root()
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
    session = backend.resolve_review_context("/tmp/multi.zarr", include_all=True)

    with np.testing.assert_raises_regex(ValueError, "not present in the current"):
        backend.apply_detection_collection(
            session,
            position=0,
            detections=[
                {"instance_key": "201", "bbox_norm": [0.4, 0.7, 0.2, 0.2], "class_id": 0}
            ],
        )


def test_apply_detection_collection_real_zarr_mints_new_key_and_preserves_survivor(
    tmp_path, monkeypatch
) -> None:
    from fisheye.shared.refined_detect_curation import write_curated_refined_detect_surfaces

    zarr_path = tmp_path / "multi_review.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update({"width": 100, "height": 80, "total_frames": 3, "recording_id": "multi-test"})
    raw = root.create_group("raw_video")
    raw.attrs["original_resolution"] = [80, 100]
    raw.create_array("images_ds", data=np.zeros((3, 80, 100), dtype=np.uint8))
    detect_parent = root.create_group("detect_runs")
    detect = detect_parent.create_group("detect_001")
    detect.create_array("instance_key", data=np.asarray([101, 102], dtype=np.uint64))
    refined_parent = root.create_group("refined_detect_runs")
    refined = refined_parent.create_group("refined_multi")
    refined.attrs["source_detect_run"] = "detect_001"

    boxes = np.asarray(
        [[0.2, 0.2, 0.2, 0.2], [0.6, 0.2, 0.2, 0.2], [0.4, 0.7, 0.2, 0.2]],
        dtype=np.float64,
    )
    write_curated_refined_detect_surfaces(
        root,
        zarr_path=zarr_path,
        refined_run_name="refined_multi",
        instance_frame_indices=np.asarray([0, 0, 2], dtype=np.int32),
        instance_bbox_norm_coords=boxes,
        instance_source_kind_labels=["raw_detect", "raw_detect", "manual"],
        instance_reason_labels=["clean", "clean", "manual_correction"],
        instance_source_detect_row_index=np.asarray([0, 1, -1], dtype=np.int32),
        instance_manual_edit_flags=np.asarray([False, False, True], dtype=bool),
        instance_confidence_scores=np.asarray([0.9, 0.8, 1.0], dtype=np.float32),
        instance_class_ids=np.asarray([0, 0, 0], dtype=np.int32),
        source_detection_source_detect_row_index=np.asarray([0, 1], dtype=np.int32),
        source_detection_frame_indices=np.asarray([0, 0], dtype=np.int32),
        source_detection_bbox_norm_coords=boxes[:2],
        source_detection_decision_labels=["accepted", "accepted"],
        source_detection_reason_labels=["clean", "clean"],
        source_detection_confidence_scores=np.asarray([0.9, 0.8], dtype=np.float32),
        source_detection_class_ids=np.asarray([0, 0], dtype=np.int32),
        source_detection_instance_key=np.asarray([101, 102], dtype=np.uint64),
    )
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(backend.detect_review_mod, "get_environment_info", lambda **_kwargs: {})

    session = backend.resolve_review_context(
        str(zarr_path), refined_run="refined_multi", include_all=True
    )
    result = backend.apply_detection_collection(
        session,
        position=0,
        detections=[
            {"instance_key": "101", "bbox_norm": [0.25, 0.25, 0.2, 0.2], "class_id": 0},
            {"instance_key": None, "bbox_norm": [0.8, 0.25, 0.1, 0.1], "class_id": 0},
        ],
    )

    instances = root["refined_detect_runs/refined_multi/instances"]
    frame_indices = np.asarray(instances["frame_indices"][:], dtype=np.int32)
    keys = np.asarray(instances["instance_key"][:], dtype=np.uint64)
    frame_zero_keys = keys[frame_indices == 0]
    decisions = np.asarray(
        root["refined_detect_runs/refined_multi/source_detections/decision_codes"][:],
        dtype=np.int8,
    )

    assert result["added"] == 1
    assert result["updated"] == 1
    assert result["removed"] == 1
    assert int(frame_zero_keys[0]) == 101
    assert int(frame_zero_keys[1]) not in {0, 101, 102, 201}
    assert np.unique(keys).shape[0] == keys.shape[0]
    assert decisions.tolist() == [0, 3]


def test_apply_manual_edit_writes_present_box(monkeypatch) -> None:
    root = _make_root()
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(backend.detect_review_mod, "get_environment_info", lambda **_kwargs: {})

    session = backend.resolve_review_context("/tmp/example.zarr", manual_score=0.77)
    result = backend.apply_manual_edit(session, position=0, bbox_norm=[0.5, 0.5, 0.25, 0.5])
    payload = backend.load_frame_payload(session, 0)

    assert result["action"] == "manual_correction"
    assert payload["bbox_norm"] == [0.5, 0.5, 0.25, 0.5]
    assert payload["status"]["status_label"] == "present"
    assert payload["status"]["source_kind_label"] == "manual"
    assert payload["status"]["reason_label"] == "manual_correction"
    assert payload["status"]["manual_edit"] is True
    assert payload["status"]["confidence_score"] == np.float32(0.77).item()


def test_apply_manual_clear_marks_filtered_out(monkeypatch) -> None:
    root = _make_root()
    monkeypatch.setattr(backend, "open_zarr_group_direct", lambda *_args, **_kwargs: root)
    captured: dict[str, Any] = {}

    def _capture_write(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(backend.detect_review_mod, "_write_dense_curated_edit_payload", _capture_write)
    monkeypatch.setattr(backend, "_reload_payload", lambda session: setattr(session, "payload", captured["payload"]))

    session = backend.resolve_review_context("/tmp/example.zarr", include_all=True)
    result = backend.apply_manual_edit(session, position=0, bbox_norm=None)
    payload = backend.load_frame_payload(session, 0)

    assert result["action"] == "manual_clear"
    assert captured["row_indices"].tolist() == [0]
    assert payload["bbox_norm"] is None
    assert payload["status"]["status_label"] == "filtered_out"
    assert payload["status"]["source_kind_label"] == "none"
    assert payload["status"]["reason_label"] == "manual_clear"
    assert payload["status"]["manual_edit"] is True


def test_apply_review_status_approved_sets_authoritative_refined_detect_run(tmp_path) -> None:
    zarr_path = tmp_path / "detect_review_approval.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_detect_runs")
    refined = refined_parent.create_group("refined_detect_001")
    mark_run_complete(refined, parent_group=refined_parent, run_name="refined_detect_001")
    session = backend.DetectReviewSession(
        zarr_path=str(zarr_path),
        root=root,
        refined_run=refined,
        refined_run_name="refined_detect_001",
        images=np.zeros((0, 1, 1), dtype=np.uint8),
        payload={},
        review_rows=np.zeros((0,), dtype=np.int32),
        total_frames=0,
        height=1,
        width=1,
        source_height=1,
        source_width=1,
        downsample_preserve_aspect=False,
        manual_score=1.0,
        manual_class_id=0,
    )

    result = backend.apply_review_status(
        session,
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="detect approved",
    )

    assert result["changed"] is True
    assert result["authoritative_approval"]["status"] == "ok"
    assert refined.attrs["detect_review_status"]["state"] == "approved"
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    reopened_parent = reopened["refined_detect_runs"]
    assert "detect_review_status_latest" not in reopened_parent.attrs
    assert reopened_parent.attrs[AUTHORITATIVE_RUN_ATTR] == "refined_detect_001"


def test_apply_review_status_approved_is_fail_closed_when_authoritative_approval_fails(monkeypatch) -> None:
    root = _make_root()
    refined_parent = root["refined_detect_runs"]
    refined = refined_parent["refined_detect_001"]
    session = backend.DetectReviewSession(
        zarr_path="/tmp/missing-detect-review.zarr",
        root=root,
        refined_run=refined,
        refined_run_name="refined_detect_001",
        images=np.zeros((0, 1, 1), dtype=np.uint8),
        payload={},
        review_rows=np.zeros((0,), dtype=np.int32),
        total_frames=0,
        height=1,
        width=1,
        source_height=1,
        source_width=1,
        downsample_preserve_aspect=False,
        manual_score=1.0,
        manual_class_id=0,
    )
    monkeypatch.setattr(
        backend,
        "_approve_authoritative_refined_detect",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "status": "blocked",
            "reason_code": "RUN_INCOMPLETE",
        },
    )

    result = backend.apply_review_status(
        session,
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="detect approved",
    )

    assert result["changed"] is False
    assert result["authoritative_approval"]["status"] == "blocked"
    assert "detect_review_status" not in refined.attrs
    assert "detect_review_status_latest" not in refined_parent.attrs
