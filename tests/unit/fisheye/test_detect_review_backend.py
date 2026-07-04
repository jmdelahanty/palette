from __future__ import annotations

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
    assert reopened_parent.attrs["detect_review_status_latest"] == "refined_detect_001"
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
