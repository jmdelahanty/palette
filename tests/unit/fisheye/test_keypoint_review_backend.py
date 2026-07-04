from __future__ import annotations

import base64
import json
import threading
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from fisheye.refinement.keypoint_quality import compute_geometry_metrics, select_head_triangle_points
from fisheye.pose.metric_schema import DerivedMetricStorage, metric_schema_from_package
from fisheye.registry.db import Registry
from fisheye.shared.zarr_run_completion import AUTHORITATIVE_RUN_ATTR, mark_run_complete
from fisheye.tune import keypoint_review_backend as mod
from fisheye.tune import keypoint_review_web as web
from fisheye.utils import zarr_io


class _FakeArray:
    def __init__(self, data: Any, *, chunks: tuple[int, ...] | None = None) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.chunks = chunks

    def __getitem__(self, item: object) -> np.ndarray:
        return self._data[item]

    def __setitem__(self, item: object, value: object) -> None:
        self._data[item] = value

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:  # pragma: no cover - compatibility
        array = np.asarray(self._data)
        return array.astype(dtype) if dtype is not None else array


class _FakeGroup(dict[str, object]):
    def __init__(self, *args: Any, attrs: dict[str, object] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default: object = None) -> object:  # type: ignore[override]
        return super().get(key, default)


def _labels_for_count(keypoint_count: int) -> list[str]:
    base = ["swim_bladder", "eye_left", "eye_right", "snout_tip", "tail_tip"]
    if keypoint_count <= len(base):
        return base[:keypoint_count]
    extra = [f"extra_{idx}" for idx in range(keypoint_count - len(base))]
    return base + extra


def _build_fake_review_root(keypoint_count: int, *, include_derived: bool = False) -> tuple[_FakeGroup, _FakeGroup, _FakeGroup]:
    row_count = 3
    labels = _labels_for_count(keypoint_count)

    refined_attrs: dict[str, object] = {
        "keypoint_labels": labels,
        "summary_statistics": {"min_triangle_area": 0.0, "min_triangle_angle": 0.0, "confidence_threshold": 0.2},
    }
    if include_derived:
        refined_attrs["pose_schema"] = {
            "schema_name": "pose_schema",
            "nodes": [{"id": idx, "name": name} for idx, name in enumerate(labels[:5])],
            "skeleton_id": "classic_fish_2d",
        }

    roi_coords = np.stack(
        [
            np.full((keypoint_count, 2), [1.0, 2.0], dtype=np.float64),
            np.zeros((keypoint_count, 2), dtype=np.float64),
            np.full((keypoint_count, 2), [3.0, 1.0], dtype=np.float64),
        ],
        axis=0,
    )

    refined = _FakeGroup(
        {
            "keypoints_roi": _FakeArray(np.full((row_count, keypoint_count, 2), np.nan, dtype=np.float64), chunks=(1, keypoint_count, 2)),
            "keypoints_img": _FakeArray(np.zeros((row_count, keypoint_count, 2), dtype=np.float64)),
            "keypoints_norm": _FakeArray(np.zeros((row_count, keypoint_count, 2), dtype=np.float64)),
            "heading": _FakeArray(np.full((row_count,), np.nan, dtype=np.float64), chunks=(row_count,)),
            "confidence": _FakeArray(np.zeros((row_count,), dtype=np.float64)),
            "keypoint_confidences": _FakeArray(np.zeros((row_count, keypoint_count), dtype=np.float64)),
            "triangle_area": _FakeArray(np.zeros((row_count,), dtype=np.float64)),
            "min_angle": _FakeArray(np.zeros((row_count,), dtype=np.float64)),
            "triangle_angles": _FakeArray(np.zeros((row_count, 3), dtype=np.float64)),
            "refined_success": _FakeArray(np.array([False, True, True], dtype=bool)),
            "flip_corrected": _FakeArray(np.array([True, False, False], dtype=bool)),
            "quality_labels": _FakeArray(np.array([2, 2, 2], dtype=np.int64)),
            "confidence_valid": _FakeArray(np.array([False, False, False], dtype=bool)),
            "geometry_valid": _FakeArray(np.array([False, False, False], dtype=bool)),
            "usable_keypoints": _FakeArray(np.array([False, False, False], dtype=bool)),
            "edit_applied": _FakeArray(np.array([False, False, False], dtype=bool)),
            "heading_finite": _FakeArray(np.array([False, False, False], dtype=bool)),
            "heading_usable": _FakeArray(np.array([False, False, False], dtype=bool)),
            "detection_source": _FakeArray(np.array([0, 0, 1], dtype=np.int64)),
            "reason": _FakeArray(np.array(["", "detection_issue", "manual_correction"], dtype=object)),
        },
        attrs=refined_attrs,
    )

    crop = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.array([10, 11, 12], dtype=np.int64)),
            "roi_coordinates_full": _FakeArray(roi_coords),
            "roi_images": _FakeArray(np.arange(3 * 16 * 16 * 1, dtype=np.uint8).reshape(3, 16, 16, 1)),
            "source_refined_row_ids": _FakeArray(np.array([1000, 1001, 1002], dtype=np.int64)),
            "source_detect_row_index": _FakeArray(np.array([200, 201, 202], dtype=np.int64)),
        },
        attrs={},
    )

    refined_parent = _FakeGroup({"refined_1": refined}, attrs={"latest": "refined_1"})
    crop_parent = _FakeGroup({"crop_1": crop}, attrs={"latest": "crop_1"})
    raw = _FakeGroup({"images_ds": _FakeArray(np.zeros((1, 24, 32), dtype=np.uint8))})
    root = _FakeGroup(
        {
            "refined_keypoints_runs": refined_parent,
            "crop_runs": crop_parent,
            "raw_video": raw,
        },
        attrs={"width": 32, "height": 24},
    )

    return root, refined_parent, crop_parent


def _build_session(*, keypoint_count: int = 5, include_derived: bool = False) -> mod.ReviewSession:
    row_count = 4
    labels = _labels_for_count(keypoint_count)

    refined = _FakeGroup(
        {
            "keypoints_roi": _FakeArray(np.full((row_count, keypoint_count, 2), np.nan, dtype=np.float64)),
            "keypoints_img": _FakeArray(np.full((row_count, keypoint_count, 2), 5.0, dtype=np.float64)),
            "keypoints_norm": _FakeArray(np.full((row_count, keypoint_count, 2), 0.25, dtype=np.float64)),
            "heading": _FakeArray(np.full((row_count,), np.nan, dtype=np.float64)),
            "confidence": _FakeArray(np.zeros((row_count,), dtype=np.float64)),
            "keypoint_confidences": _FakeArray(np.zeros((row_count, keypoint_count), dtype=np.float64)),
            "triangle_area": _FakeArray(np.zeros((row_count,), dtype=np.float64)),
            "min_angle": _FakeArray(np.zeros((row_count,), dtype=np.float64)),
            "triangle_angles": _FakeArray(np.zeros((row_count, 3), dtype=np.float64)),
            "refined_success": _FakeArray(np.array([False, False, False, False], dtype=bool)),
            "flip_corrected": _FakeArray(np.array([True, True, True, True], dtype=bool)),
            "quality_labels": _FakeArray(np.array([2, 2, 2, 2], dtype=np.int64)),
            "confidence_valid": _FakeArray(np.array([False, False, False, False], dtype=bool)),
            "geometry_valid": _FakeArray(np.array([False, False, False, False], dtype=bool)),
            "usable_keypoints": _FakeArray(np.array([False, False, False, False], dtype=bool)),
            "edit_applied": _FakeArray(np.array([False, False, False, False], dtype=bool)),
            "heading_finite": _FakeArray(np.array([False, False, False, False], dtype=bool)),
            "heading_usable": _FakeArray(np.array([False, False, False, False], dtype=bool)),
            "detection_source": _FakeArray(np.array([0, 1, 0, 0], dtype=np.int64)),
            "reason": _FakeArray(np.array(["", "", "", ""], dtype=object)),
        },
        attrs={"keypoint_labels": labels, "summary_statistics": {"confidence_threshold": 0.2}},
    )
    crop = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.array([10, 11, 12, 13], dtype=np.int64)),
            "roi_coordinates_full": _FakeArray(np.arange(row_count * keypoint_count * 2, dtype=np.float64).reshape(row_count, keypoint_count, 2)),
            "roi_images": _FakeArray(np.ones((row_count, 12, 16, 1), dtype=np.uint8)),
            "source_refined_row_ids": _FakeArray(np.array([1000, 1001, 1002, 1003], dtype=np.int64)),
            "source_detect_row_index": _FakeArray(np.array([200, 201, 202, 203], dtype=np.int64)),
        }
    )
    root = _FakeGroup(
        {"refined_keypoints_runs": _FakeGroup({}), "crop_runs": _FakeGroup({})},
        attrs={"width": 16, "height": 12},
    )

    head_triangle_indices = mod.resolve_head_triangle_for_labels(
        labels,
        keypoint_count=keypoint_count,
        allow_legacy_3point_fallback=True,
    )
    source_refined_row_ids = np.array([1000, 1001, 1002, 1003], dtype=np.int64)
    source_detect_row_index = np.array([200, 201, 202, 203], dtype=np.int64)

    derived_storage: DerivedMetricStorage | None = None
    if include_derived and keypoint_count >= 5:
        schema = metric_schema_from_package("traditional_v2")
        metric_count = len(schema.metrics)
        derived_storage = DerivedMetricStorage(
            schema=schema,
            values=_FakeArray(np.full((row_count, metric_count), np.nan, dtype=np.float32)),
            values_norm=_FakeArray(np.full((row_count, metric_count), np.nan, dtype=np.float32)),
            valid=_FakeArray(np.zeros((row_count, metric_count), dtype=bool)),
        )

    return mod.ReviewSession(
        zarr_path="in-memory",
        root=root,
        refined=refined,
        crop=crop,
        refined_run="refined_1",
        crop_run="crop_1",
        failures=np.asarray([0, 1, 2, 3], dtype="i4"),
        frame_indices=np.asarray([10, 11, 12, 13], dtype=np.int64),
        roi_images=crop["roi_images"],
        roi_coordinates_full=crop["roi_coordinates_full"],
        source_refined_row_ids=source_refined_row_ids,
        source_detect_row_index=source_detect_row_index,
        keypoint_labels=labels,
        keypoint_count=keypoint_count,
        roi_diagonal=20.0,
        norm_factor=np.array([16, 12], dtype=np.float64),
        kp_roi_arr=refined["keypoints_roi"],
        kp_img_arr=refined["keypoints_img"],
        kp_norm_arr=refined["keypoints_norm"],
        heading_arr=refined["heading"],
        confidence_arr=refined["confidence"],
        conf_arr=refined["keypoint_confidences"],
        triangle_area_arr=refined["triangle_area"],
        min_angle_arr=refined["min_angle"],
        triangle_angles_arr=refined["triangle_angles"],
        refined_success_arr=refined["refined_success"],
        flip_corrected_arr=refined["flip_corrected"],
        quality_labels_arr=refined["quality_labels"],
        confidence_valid_arr=refined["confidence_valid"],
        geometry_valid_arr=refined["geometry_valid"],
        usable_arr=refined["usable_keypoints"],
        edit_applied_arr=refined["edit_applied"],
        reason_arr=refined["reason"],
        heading_finite_arr=refined["heading_finite"],
        heading_usable_arr=refined["heading_usable"],
        detection_source_arr=refined["detection_source"],
        min_triangle_angle=0.0,
        min_triangle_area=0.0,
        max_triangle_area=None,
        confidence_threshold=0.2,
        head_triangle_indices=head_triangle_indices,
        derived_metric_storage=derived_storage,
    )


def _make_keypoint_reviewable_zarr_shell(path: Path) -> None:
    for child in ("refined_keypoints_runs", "crop_runs"):
        child_path = path / child
        child_path.mkdir(parents=True, exist_ok=True)
        (child_path / "zarr.json").write_text("{}", encoding="utf-8")


def _mark_keypoint_review_approved(path: Path, *, run_name: str = "refined_1") -> None:
    parent = path / "refined_keypoints_runs"
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"latest": run_name}}),
        encoding="utf-8",
    )
    run = parent / run_name
    run.mkdir(parents=True, exist_ok=True)
    (run / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "keypoint_review_status": {
                        "state": "approved",
                        "method": "manual",
                        "intended_use": "training",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_list_review_rois_filters_include_all_and_targets() -> None:
    _, refined_parent, _ = _build_fake_review_root(3)
    refined = refined_parent["refined_1"]

    failures, targeted = mod.list_review_rois(
        refined,
        np.array([10, 11, 12], dtype=np.int64),
        include_all=False,
        target_frames=None,
        target_roi_indices=None,
    )
    np.testing.assert_array_equal(failures, np.array([0], dtype="i4"))
    assert targeted is False

    failures_targeted = mod.list_review_rois(
        refined,
        np.array([10, 11, 12], dtype=np.int64),
        include_all=False,
        target_frames=[12],
        target_roi_indices=[0],
    )
    np.testing.assert_array_equal(failures_targeted[0], np.array([0, 2], dtype="i4"))
    assert failures_targeted[1] is True

    all_failures, all_targeted = mod.list_review_rois(
        refined,
        np.array([10, 11, 12], dtype=np.int64),
        include_all=True,
        target_frames=None,
        target_roi_indices=None,
    )
    np.testing.assert_array_equal(all_failures, np.array([0, 1, 2], dtype="i4"))
    assert all_targeted is False


def test_resolve_latest_refined_and_crop_uses_latest_runs() -> None:
    root, refined_parent, crop_parent = _build_fake_review_root(3)
    expected_refined = refined_parent["refined_1"]
    expected_crop = crop_parent["crop_1"]

    monkeypatch = pytest.MonkeyPatch()
    original = mod.open_zarr_root
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    try:
        resolved_root, resolved_refined, resolved_crop, resolved_refined_run, resolved_crop_run = mod.resolve_latest_refined_and_crop(
            "in-memory",
            mode="a",
        )
        assert resolved_root is root
        assert resolved_refined is expected_refined
        assert resolved_crop is expected_crop
        assert resolved_refined_run == "refined_1"
        assert resolved_crop_run == "crop_1"
    finally:
        mod.open_zarr_root = original
        monkeypatch.undo()


def test_resolve_review_session_targets_frame_and_roi_indices() -> None:
    root, refined_parent, _ = _build_fake_review_root(3)

    monkeypatch = pytest.MonkeyPatch()
    original = mod.open_zarr_root
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    try:
        session = mod.resolve_review_session(
            "in-memory",
            include_all=False,
            target_frames=[12],
            target_roi_indices=[0],
        )
    finally:
        mod.open_zarr_root = original
        monkeypatch.undo()

    np.testing.assert_array_equal(session.failures, np.array([0, 2], dtype="i4"))


def test_open_group_uses_non_consolidated_for_mutable_root_calls(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(zarr_io, "LocalStore", None)

    class _FakeRoot:
        def __init__(self) -> None:
            self.attrs: dict[str, object] = {}

    def _fake_open_group(*_args: Any, **kwargs: Any) -> _FakeRoot:
        calls.append(kwargs)
        return _FakeRoot()

    monkeypatch.setattr(zarr_io.zarr, "open_group", _fake_open_group)

    root = zarr_io.open_zarr_root("/tmp/ignored.zarr", mode="a")
    assert root is not None
    assert any(
        kw.get("use_consolidated") is False or kw.get("consolidated") is False for kw in calls
    )


def test_load_roi_payload_includes_points_and_image_metadata() -> None:
    session = _build_session(keypoint_count=3)
    payload = mod.load_roi_payload(session, position=1)

    assert payload["roi_idx"] == 1
    assert payload["position"] == 1
    assert payload["total"] == 4
    assert payload["frame_idx"] == 11
    assert len(payload["labels"]) == 3

    image_payload = payload["roi_image"]
    assert image_payload["shape"] == [12, 16, 1]
    assert image_payload["dtype"] == "uint8"
    assert image_payload["encoding"] == "base64_raw"

    raw = session.roi_images[1]
    decoded = base64.b64decode(payload["roi_image"]["pixels"])
    assert decoded == raw.tobytes()


def test_load_roi_payload_encodes_missing_points_as_valid_json_nulls() -> None:
    session = _build_session(keypoint_count=5)
    payload = mod.load_roi_payload(session, position=0)

    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text
    assert payload["points"][0] == [None, None]


@pytest.mark.parametrize("keypoint_count", (3, 5))
def test_save_roi_correction_updates_keypoint_rows_and_flags(keypoint_count: int) -> None:
    session = _build_session(keypoint_count=keypoint_count, include_derived=(keypoint_count == 5))
    points = np.linspace(
        1.0,
        1.0 + (keypoint_count - 1) * 2.0,
        num=keypoint_count * 2,
        dtype=np.float64,
    ).reshape(keypoint_count, 2)

    result = mod.save_roi_correction(session, position=0, points=points)

    np.testing.assert_allclose(np.asarray(session.kp_roi_arr[0]), points)
    np.testing.assert_allclose(np.asarray(session.kp_img_arr[0]), points + np.asarray(session.roi_coordinates_full[0], dtype=np.float64))
    np.testing.assert_allclose(
        np.asarray(session.kp_norm_arr[0]),
        (points + np.asarray(session.roi_coordinates_full[0], dtype=np.float64)) / session.norm_factor,
    )

    assert bool(session.refined_success_arr[0]) is True
    assert bool(session.confidence_valid_arr[0]) is True
    assert bool(session.geometry_valid_arr[0]) == bool(result["geometry_ok"])
    assert bool(session.usable_arr[0]) == (bool(result["confidence_ok"]) and bool(result["geometry_ok"]))
    assert bool(session.edit_applied_arr[0]) is True
    expected_heading_finite = result["heading"] is not None and bool(np.isfinite(result["heading"]))
    assert bool(session.heading_finite_arr[0]) == expected_heading_finite
    assert "manual_correction" in str(session.reason_arr[0])
    assert result["changed"] is True
    assert result["stale_touched"] >= 0

    if session.derived_metric_storage is not None:
        assert bool(np.asarray(session.derived_metric_storage.valid[0], dtype=bool).any())


def test_save_roi_correction_no_change_does_not_mark_stale_or_edit() -> None:
    session = _build_session(keypoint_count=5)
    base_points = np.array(
        [
            [2.0, 1.0],
            [4.0, 2.0],
            [6.0, 5.0],
            [8.0, 6.0],
            [10.0, 7.0],
        ],
        dtype=np.float64,
    )
    roi_coordinates = np.asarray(session.roi_coordinates_full[0], dtype=np.float64)
    full_points = base_points + roi_coordinates
    session.kp_roi_arr[0] = base_points.copy()
    session.kp_img_arr[0] = full_points.copy()
    session.kp_norm_arr[0] = full_points / session.norm_factor

    heading = mod.compute_heading_from_attrs(
        session.refined.attrs,
        labels=session.keypoint_labels,
        points=base_points,
    )
    session.heading_arr[0] = heading

    metrics = compute_geometry_metrics(select_head_triangle_points(base_points, session.head_triangle_indices))
    geometry_ok = bool(
        np.isfinite(metrics.min_angle)
        and np.isfinite(metrics.area)
        and metrics.min_angle >= float(session.min_triangle_angle)
        and metrics.area >= float(session.min_triangle_area)
    )
    session.triangle_area_arr[0] = metrics.area
    session.min_angle_arr[0] = metrics.min_angle
    session.triangle_angles_arr[0] = metrics.angles

    session.conf_arr[0] = np.ones(session.keypoint_count, dtype=np.float64)
    session.confidence_arr[0] = 1.0
    session.refined_success_arr[0] = True
    session.flip_corrected_arr[0] = False
    session.quality_labels_arr[0] = 0
    session.confidence_valid_arr[0] = True
    session.geometry_valid_arr[0] = geometry_ok
    session.usable_arr[0] = geometry_ok
    session.heading_finite_arr[0] = bool(np.isfinite(heading))
    session.heading_usable_arr[0] = bool(np.isfinite(heading))
    session.reason_arr[0] = mod._build_manual_reason("", geom_ok=geometry_ok)
    session.edit_applied_arr[0] = True

    touched: dict[str, object] = {}

    def _fake_mark(*_: Any, **kwargs: object) -> int:
        touched.update(kwargs)
        return 7

    original_mark = mod.mark_downstream_subject_mask_runs_stale
    original = mod.ReviewSession
    mod.mark_downstream_subject_mask_runs_stale = _fake_mark  # type: ignore[assignment]
    try:
        result = mod.save_roi_correction(session, position=0, points=base_points)
    finally:
        mod.mark_downstream_subject_mask_runs_stale = original_mark
        mod.ReviewSession = original

    assert result["changed"] is False
    assert result["stale_touched"] == 0
    assert touched == {}
    assert bool(result["reason_updated"]) is False
    assert bool(session.edit_applied_arr[0]) is True
    np.testing.assert_array_equal(session.source_refined_row_ids, np.array([1000, 1001, 1002, 1003], dtype=np.int64))
    np.testing.assert_array_equal(session.source_detect_row_index, np.array([200, 201, 202, 203], dtype=np.int64))


def test_mark_no_keypoints_updates_failure_state_and_reason(monkeypatch) -> None:
    session = _build_session(keypoint_count=5)
    points = np.ones((5, 2), dtype=np.float64)
    session.kp_roi_arr[0] = points
    stale_calls: list[dict[str, object]] = []

    def _fake_mark(*_: Any, **kwargs: object) -> int:
        stale_calls.append(dict(kwargs))
        return 3

    monkeypatch.setattr(mod, "mark_downstream_subject_mask_runs_stale", _fake_mark)

    result = mod.mark_no_keypoints(session, position=0)

    assert result["action"] == "mark_no_keypoints"
    assert result["changed"] is True
    assert result["stale_touched"] == 3
    assert np.isnan(np.asarray(session.kp_roi_arr[0], dtype=float)).all()
    assert bool(session.refined_success_arr[0]) is False
    assert bool(session.usable_arr[0]) is False
    assert bool(session.edit_applied_arr[0]) is True
    assert "fish_present_no_keypoints" in str(session.reason_arr[0])
    assert stale_calls[0]["reason"] == "keypoint_mark_no_keypoints"


def test_mark_detection_issue_and_clear_failure_label(monkeypatch) -> None:
    session = _build_session(keypoint_count=5)
    monkeypatch.setattr(mod, "mark_downstream_subject_mask_runs_stale", lambda *_args, **_kwargs: 1)

    result = mod.mark_detection_issue(session, position=0)
    assert result["action"] == "mark_detection_issue"
    assert "detection_issue" in str(session.reason_arr[0])
    assert bool(session.refined_success_arr[0]) is False

    clear = mod.clear_failure_label(session, position=0)
    assert clear["action"] == "clear_failure_label"
    assert clear["changed"] is True
    assert "detection_issue" not in str(session.reason_arr[0])
    assert "manual_correction" in str(session.reason_arr[0])


def test_filter_review_rois_supports_failed_all_manual_and_search() -> None:
    session = _build_session(keypoint_count=5)
    session.refined_success_arr[0] = False
    session.refined_success_arr[1] = True
    session.refined_success_arr[2] = False
    session.refined_success_arr[3] = True
    session.reason_arr[0] = "low_confidence"
    session.reason_arr[1] = "manual_correction"
    session.reason_arr[2] = "detection_issue"
    session.edit_applied_arr[1] = True

    np.testing.assert_array_equal(mod.filter_review_rois(session, filter_mode="failed"), np.array([0], dtype="i4"))
    np.testing.assert_array_equal(mod.filter_review_rois(session, filter_mode="all"), np.array([0, 1, 2, 3], dtype="i4"))
    np.testing.assert_array_equal(mod.filter_review_rois(session, filter_mode="manual"), np.array([1], dtype="i4"))
    np.testing.assert_array_equal(mod.filter_review_rois(session, filter_mode="edited"), np.array([1], dtype="i4"))
    np.testing.assert_array_equal(mod.filter_review_rois(session, filter_mode="all", search="frame=12"), np.array([2], dtype="i4"))


def test_flag_followup_frame_writes_identity_payload(tmp_path) -> None:
    session = _build_session(keypoint_count=5)
    flag_path = tmp_path / "flags.json"

    result = mod.flag_followup_frame(session, position=0, flag_path=flag_path)

    assert result["action"] == "flag_followup"
    data = json.loads(flag_path.read_text(encoding="utf-8"))
    entries = data["in-memory"]
    assert entries[0]["frame_idx"] == 10
    assert entries[0]["roi_idx"] == 0
    assert entries[0]["source_refined_row_id"] == 1000
    assert entries[0]["source_detect_row_index"] == 200


def test_apply_review_status_delegates_existing_status_writer(monkeypatch) -> None:
    session = _build_session(keypoint_count=5)
    parent = _FakeGroup({"refined_1": session.refined}, attrs={})
    session.root["refined_keypoints_runs"] = parent

    def _fake_apply(refined_parent: object, refined_run: str, refined: object, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        payload = {
            "state": kwargs["state"],
            "method": kwargs["method"],
            "intended_use": kwargs["intended_use"],
            "reviewer": kwargs["reviewer"],
            "notes": kwargs["notes"],
        }
        refined.attrs["keypoint_review_status"] = payload
        refined_parent.attrs["keypoint_review_status_latest"] = refined_run
        return payload, {"synced": True}

    monkeypatch.setattr(mod, "_apply_review_status", _fake_apply)
    monkeypatch.setattr(mod, "_update_postprocess_summary", lambda refined, *, root=None, print_summary=False: {"total_rois": 4})

    result = mod.apply_review_status(
        session,
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="ok",
    )

    assert result["review_status"]["state"] == "approved"
    assert result["review_status"]["intended_use"] == "training"
    assert result["postprocess_summary"]["total_rois"] == 4
    assert parent.attrs["keypoint_review_status_latest"] == "refined_1"
    assert result["authoritative_approval"]["attempted"] is False


def test_apply_review_status_approved_sets_authoritative_refined_keypoint_run(monkeypatch, tmp_path) -> None:
    zarr_path = tmp_path / "keypoint_review_approval.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    refined_parent = root.create_group("refined_keypoints_runs")
    refined = refined_parent.create_group("refined_1")
    mark_run_complete(refined, parent_group=refined_parent, run_name="refined_1")

    session = _build_session(keypoint_count=5)
    session.zarr_path = str(zarr_path)
    session.root = root
    session.refined = refined
    monkeypatch.setattr(mod, "_update_postprocess_summary", lambda refined, *, root=None, print_summary=False: {"total_rois": 4})

    result = mod.apply_review_status(
        session,
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="keypoints approved",
    )

    assert result["review_status"]["state"] == "approved"
    assert result["authoritative_approval"]["attempted"] is True
    assert result["authoritative_approval"]["status"] == "ok"
    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert reopened["refined_keypoints_runs"].attrs[AUTHORITATIVE_RUN_ATTR] == "refined_1"


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_web_endpoints_cover_action_filter_jump_and_review_status(monkeypatch) -> None:
    session = _build_session(keypoint_count=5)
    parent = _FakeGroup({"refined_1": session.refined}, attrs={})
    session.root["refined_keypoints_runs"] = parent
    state = web._ServerState(
        session=session,
        position=0,
        filter_mode="failed",
        review_method="manual",
        review_intended_use="training",
        reviewer="tester",
    )

    monkeypatch.setattr(mod, "mark_downstream_subject_mask_runs_stale", lambda *_args, **_kwargs: 0)

    def _fake_apply(refined_parent: object, refined_run: str, refined: object, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        payload = {"state": kwargs["state"], "method": kwargs["method"], "intended_use": kwargs["intended_use"]}
        refined.attrs["keypoint_review_status"] = payload
        return payload, {"synced": True}

    monkeypatch.setattr(mod, "_apply_review_status", _fake_apply)
    monkeypatch.setattr(mod, "_update_postprocess_summary", lambda refined, *, root=None, print_summary=False: {"total_rois": 4})

    handler = web._build_handler(state=state, static_root=Path(__file__).parent, backend_module=mod)
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        action = _post_json(f"{base}/api/roi/current/action", {"action": "mark_no_keypoints"})
        assert action["ok"] is True
        assert action["result"]["action"] == "mark_no_keypoints"

        filtered = _post_json(f"{base}/api/filter", {"filter_mode": "all", "search": "frame=12"})
        assert filtered["ok"] is True
        assert filtered["state"]["total"] == 1

        jumped = _post_json(f"{base}/api/jump", {"roi_idx": 0})
        assert jumped["ok"] is True
        assert jumped["state"]["roi_idx"] == 0

        status = _post_json(f"{base}/api/review_status", {"state": "approved"})
        assert status["ok"] is True
        assert status["result"]["review_status"]["state"] == "approved"

        current = _get_json(f"{base}/api/roi/current")
        text = json.dumps(current, allow_nan=False)
        assert "NaN" not in text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_registry_training_dataset_listing_filters_to_training_zarrs(tmp_path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    registry = Registry(registry_path)
    _make_keypoint_reviewable_zarr_shell(tmp_path / "train.zarr")
    registry.upsert_dataset(
        "train_ds",
        session_uuid="train_session",
        zarr_path=tmp_path / "train.zarr",
        recording_id="rec_train",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "analysis_ds",
        session_uuid="analysis_session",
        zarr_path=tmp_path / "analysis.zarr",
        recording_id="rec_analysis",
        zarr_use="analysis",
    )

    rows = web._list_training_datasets(str(registry_path), limit=50)

    assert [row["dataset_id"] for row in rows] == ["train_ds"]
    assert rows[0]["zarr_path"] == str(tmp_path / "train.zarr")
    assert rows[0]["recording_id"] == "rec_train"


def test_registry_training_dataset_listing_defaults_to_unapproved_keypoint_zarrs(tmp_path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    registry = Registry(registry_path)
    approved_zarr = tmp_path / "approved.zarr"
    needs_review_zarr = tmp_path / "needs_review.zarr"
    _make_keypoint_reviewable_zarr_shell(approved_zarr)
    _mark_keypoint_review_approved(approved_zarr)
    _make_keypoint_reviewable_zarr_shell(needs_review_zarr)
    registry.upsert_dataset(
        "approved_ds",
        session_uuid="approved",
        zarr_path=approved_zarr,
        recording_id="rec_approved",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "needs_review_ds",
        session_uuid="needs_review",
        zarr_path=needs_review_zarr,
        recording_id="rec_needs_review",
        zarr_use="training",
    )

    default_rows = web._list_training_datasets(str(registry_path), limit=50)
    all_rows = web._list_training_datasets(str(registry_path), limit=50, review_filter="all")
    approved_rows = web._list_training_datasets(str(registry_path), limit=50, review_filter="approved")

    assert [row["dataset_id"] for row in default_rows] == ["needs_review_ds"]
    assert {row["dataset_id"] for row in all_rows} == {"approved_ds", "needs_review_ds"}
    assert [row["dataset_id"] for row in approved_rows] == ["approved_ds"]
    assert approved_rows[0]["keypoint_review_approved"] is True


def test_web_registry_endpoints_list_and_switch_training_dataset(monkeypatch, tmp_path) -> None:
    registry_path = tmp_path / "palette_registry.sqlite"
    registry = Registry(registry_path)
    zarr_a = tmp_path / "train_a.zarr"
    zarr_b = tmp_path / "train_b.zarr"
    _make_keypoint_reviewable_zarr_shell(zarr_a)
    _make_keypoint_reviewable_zarr_shell(zarr_b)
    registry.upsert_dataset(
        "train_a",
        session_uuid="train_a",
        zarr_path=zarr_a,
        recording_id="recording_a",
        zarr_use="training",
    )
    registry.upsert_dataset(
        "train_b",
        session_uuid="train_b",
        zarr_path=zarr_b,
        recording_id="recording_b",
        zarr_use="training",
    )

    initial_session = _build_session(keypoint_count=5)
    initial_session.zarr_path = str(zarr_a)
    state = web._ServerState(
        session=initial_session,
        position=0,
        filter_mode="failed",
        registry_path=str(registry_path),
        dataset_id="train_a",
        dataset={"dataset_id": "train_a", "zarr_path": str(zarr_a)},
        lock_enabled=False,
    )

    def _fake_resolve_review_session(zarr_path: str, **_: object) -> mod.ReviewSession:
        session = _build_session(keypoint_count=5)
        session.zarr_path = str(zarr_path)
        session.refined_run = f"refined_for_{Path(zarr_path).stem}"
        return session

    monkeypatch.setattr(mod, "resolve_review_session", _fake_resolve_review_session)

    handler = web._build_handler(state=state, static_root=Path(__file__).parent, backend_module=mod)
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        listed = _get_json(f"{base}/api/registry/datasets")
        assert listed["ok"] is True
        assert listed["enabled"] is True
        assert [row["dataset_id"] for row in listed["datasets"]] == ["train_a", "train_b"]

        selected = _post_json(f"{base}/api/registry/select", {"dataset_id": "train_b"})
        assert selected["ok"] is True
        assert selected["state"]["dataset_id"] == "train_b"
        assert selected["state"]["zarr_path"] == str(zarr_b)
        assert selected["state"]["refined_run"] == "refined_for_train_b"
        assert selected["state"]["dataset_summary"]["total_rois"] == 4

        current = _get_json(f"{base}/api/roi/current")
        assert current["ok"] is True
        assert current["state"]["dataset_id"] == "train_b"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_web_config_ignores_whitespace_positional_zarr_in_registry_mode(tmp_path) -> None:
    args = web.parse_args(
        [
            " ",
            "--registry",
            str(tmp_path / "palette_registry.sqlite"),
            "--manual",
        ]
    )

    config = web.build_server_config(args)

    assert config.zarr_path is None
    assert config.registry_path == str(tmp_path / "palette_registry.sqlite")
