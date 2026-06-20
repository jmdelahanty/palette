from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

import fisheye.visualization.visualize_swim_bladder_mask_patches as swim_mod
from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.visualization.visualize_swim_bladder_mask_patches import (
    _build_view,
    _extract_patch_bounds,
    _mouse_modifier_state,
    _pad_canvas_to_shape,
    _resolve_keypoint_group,
    _resolve_erase_mode,
    _resolve_swim_bladder_center_with_source,
    _resolve_swim_bladder_keypoint_center,
    _source_component_mask_row,
    _target_canvas_shape,
    _validate_keypoint_group_alignment,
    parse_args,
)


class _FakeArray:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, key):
        return self._data[key]


class _FakeGroup(dict):
    def __init__(self, *args, attrs: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attrs = attrs or {}

    def get(self, key: str, default=None):
        return super().get(key, default)


def test_extract_patch_bounds_clamps_at_edges() -> None:
    x0, x1, y0, y1 = _extract_patch_bounds((20, 20), (1.2, 2.8), padding=5)
    assert (x0, x1, y0, y1) == (0, 7, 0, 9)


def test_resolve_swim_bladder_center_prefers_label_then_mask_centroid_then_roi_center() -> None:
    keypoints = np.array(
        [
            [10.0, 11.0],  # eye_left
            [20.0, 21.0],  # eye_right
            [3.0, 4.0],    # swim_bladder
        ],
        dtype=np.float32,
    )
    labels = ["eye_left", "eye_right", "swim_bladder"]
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[6, 7] = 1

    center_keypoint, source_keypoint = _resolve_swim_bladder_center_with_source(keypoints, labels, mask, (12, 12))
    assert center_keypoint == (3.0, 4.0)
    assert source_keypoint == "keypoint"

    keypoints_missing = np.full((3, 2), np.nan, dtype=np.float32)
    mask_centroid = np.zeros((12, 12), dtype=np.uint8)
    mask_centroid[2, 4] = 1
    mask_centroid[4, 6] = 1
    center_mask, source_mask = _resolve_swim_bladder_center_with_source(
        keypoints_missing,
        labels,
        mask_centroid,
        (12, 12),
    )
    assert center_mask == (5.0, 3.0)
    assert source_mask == "mask_centroid"

    empty_mask = np.zeros((10, 14), dtype=np.uint8)
    center_roi, source_roi = _resolve_swim_bladder_center_with_source(
        keypoints_missing,
        labels,
        empty_mask,
        (10, 14),
    )
    assert center_roi == (7.0, 5.0)
    assert source_roi == "roi_center"


def test_parse_args_sets_defaults() -> None:
    args = parse_args(["/tmp/example.zarr"])
    assert args.padding == 18
    assert args.scale_percent == 220
    assert args.edit_zoom == 8
    assert args.roi_indices is None
    assert args.debug_ui_log is None
    assert args.review_state == "approved"
    assert args.review_method == "manual"
    assert args.review_intended_use == "training"


def test_parse_args_parses_roi_indices_csv() -> None:
    args = parse_args(["/tmp/example.zarr", "--roi-indices", "4, 10,12"])
    assert args.roi_indices == [4, 10, 12]


def test_parse_args_parses_debug_ui_log() -> None:
    args = parse_args(["/tmp/example.zarr", "--debug-ui-log", "/tmp/swim-ui.jsonl"])
    assert args.debug_ui_log == Path("/tmp/swim-ui.jsonl")


def test_resolve_swim_bladder_keypoint_center_fails_closed_for_missing_or_unsuccessful() -> None:
    keypoints = np.asarray([[[3.0, 4.0], [8.0, 8.0]]], dtype=np.float32)[0]
    labels = ["swim_bladder", "eye_left"]

    center_ok, source_ok = _resolve_swim_bladder_keypoint_center(
        keypoints,
        labels,
        success_flag=True,
    )
    assert center_ok == (3.0, 4.0)
    assert source_ok == "keypoint"

    center_bad, source_bad = _resolve_swim_bladder_keypoint_center(
        keypoints,
        labels,
        success_flag=False,
    )
    assert center_bad is None
    assert source_bad == "unsuccessful_keypoint"

    center_missing, source_missing = _resolve_swim_bladder_keypoint_center(
        np.full_like(keypoints, np.nan),
        labels,
        success_flag=True,
    )
    assert center_missing is None
    assert source_missing == "missing_keypoint"


def test_source_component_mask_row_prefers_source_abstraction_over_group_array() -> None:
    source_masks = np.zeros((2, 1, 8, 8), dtype=np.uint8)
    source_masks[1, 0, 2:4, 3:5] = 1
    stale_group_masks = np.zeros_like(source_masks)
    source = SimpleNamespace(
        masks_roi=_FakeArray(source_masks),
        group=_FakeGroup({"masks_roi": _FakeArray(stale_group_masks)}),
        mask_labels=["swim_bladder"],
        available_channels=np.asarray([True], dtype=bool),
    )

    mask = _source_component_mask_row(source, "swim_bladder", 1, fallback_shape=(8, 8))

    np.testing.assert_array_equal(mask, source_masks[1, 0])


def test_source_component_mask_row_uses_zero_fallback_for_unavailable_component() -> None:
    source = SimpleNamespace(
        masks_roi=None,
        group=_FakeGroup({}),
        mask_labels=["swim_bladder"],
        available_channels=np.asarray([False], dtype=bool),
    )

    mask = _source_component_mask_row(source, "swim_bladder", 0, fallback_shape=(6, 7))

    np.testing.assert_array_equal(mask, np.zeros((6, 7), dtype=np.uint8))


def test_source_component_mask_row_reads_compact_mask_store(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "compact_source.zarr"), mode="w")
    run = root.create_group("subject_mask_runs").create_group("subject_masks_001")
    labels = ["subject_body", "swim_bladder"]
    run.attrs["mask_labels"] = labels
    masks = np.zeros((2, 2, 8, 8), dtype=np.uint8)
    masks[1, 1, 2:4, 3:6] = 1
    dense = run.create_array("masks_roi", data=masks, chunks=(1, 1, 8, 8), overwrite=True)
    run.create_array("available_channels", data=np.asarray([True, True], dtype=bool), overwrite=True)
    write_component_rle_mask_store_from_dense(
        run,
        dense,
        component_names=labels,
        encode_row_chunk_size=1,
    )
    del run["masks_roi"]
    source = SimpleNamespace(
        masks_roi=None,
        group=run,
        mask_labels=labels,
        available_channels=np.asarray([True, True], dtype=bool),
    )

    mask = _source_component_mask_row(source, "swim_bladder", 1, fallback_shape=(8, 8))

    np.testing.assert_array_equal(mask, masks[1, 1])
    assert "masks_roi" not in run


def test_validate_keypoint_group_alignment_rejects_mismatched_crop_lineage() -> None:
    crop_group = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.asarray([0, 1], dtype=np.int32)),
            "detection_indices": _FakeArray(np.asarray([10, 11], dtype=np.int32)),
        }
    )
    keypoint_group = _FakeGroup(
        {
            "keypoints_roi": _FakeArray(np.zeros((2, 5, 2), dtype=np.float32)),
            "frame_indices": _FakeArray(np.asarray([0, 1], dtype=np.int32)),
            "detection_indices": _FakeArray(np.asarray([10, 11], dtype=np.int32)),
        },
        attrs={"source_crop_run": "crop_999"},
    )

    with pytest.raises(RuntimeError, match="source_crop_run"):
        _validate_keypoint_group_alignment(
            crop_group,
            "crop_001",
            keypoint_group,
            total_rois=2,
        )


def test_validate_keypoint_group_alignment_rejects_detection_index_mismatch() -> None:
    crop_group = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.asarray([0, 1], dtype=np.int32)),
            "detection_indices": _FakeArray(np.asarray([10, 11], dtype=np.int32)),
        }
    )
    keypoint_group = _FakeGroup(
        {
            "keypoints_roi": _FakeArray(np.zeros((2, 5, 2), dtype=np.float32)),
            "frame_indices": _FakeArray(np.asarray([0, 1], dtype=np.int32)),
            "detection_indices": _FakeArray(np.asarray([10, 12], dtype=np.int32)),
        },
        attrs={"source_crop_run": "crop_001"},
    )

    with pytest.raises(RuntimeError, match="detection_indices"):
        _validate_keypoint_group_alignment(
            crop_group,
            "crop_001",
            keypoint_group,
            total_rois=2,
        )


def test_resolve_keypoint_group_prefers_run_matching_expected_crop() -> None:
    root = _FakeGroup(
        {
            "refined_keypoints_runs": _FakeGroup(
                {
                    "refined_keypoints_001": _FakeGroup(
                        {"keypoints_roi": _FakeArray(np.zeros((2, 5, 2), dtype=np.float32))},
                        attrs={"source_crop_run": "crop_001"},
                    ),
                    "refined_keypoints_002": _FakeGroup(
                        {"keypoints_roi": _FakeArray(np.zeros((2, 5, 2), dtype=np.float32))},
                        attrs={"source_crop_run": "crop_999"},
                    ),
                },
                attrs={"latest": "refined_keypoints_002"},
            )
        }
    )

    group, group_name, run_name = _resolve_keypoint_group(
        root,
        subject_group=None,
        refined_group=None,
        explicit_run=None,
        explicit_group=None,
        expected_crop_run="crop_001",
    )

    assert group is root["refined_keypoints_runs"]["refined_keypoints_001"]
    assert group_name == "refined_keypoints_runs"
    assert run_name == "refined_keypoints_001"


def test_mouse_modifier_state_decodes_ctrl_shift_lmb() -> None:
    cv2 = pytest.importorskip("cv2")
    flags = int(cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_SHIFTKEY | cv2.EVENT_FLAG_LBUTTON)
    ctrl, shift, lmb = _mouse_modifier_state(flags)
    assert ctrl is True
    assert shift is True
    assert lmb is True


def test_resolve_erase_mode_allows_shift_temporary_inverse() -> None:
    assert _resolve_erase_mode(False, False) is False
    assert _resolve_erase_mode(True, False) is True
    assert _resolve_erase_mode(False, True) is True
    assert _resolve_erase_mode(True, True) is False


def test_create_viewer_uses_crop_source_group_for_alignment(monkeypatch, tmp_path) -> None:
    class _StopViewer(RuntimeError):
        pass

    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.total_rois = 1

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            return np.zeros((max(0, stop - start), 8, 8), dtype=np.uint8)

    root = _FakeGroup()
    crop_group = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.asarray([0], dtype=np.int32)),
            "detection_indices": _FakeArray(np.asarray([10], dtype=np.int32)),
        }
    )
    crop_source = _FakeCropSource(crop_group)
    keypoint_group = _FakeGroup(
        {
            "keypoints_roi": _FakeArray(np.zeros((1, 1, 2), dtype=np.float32)),
            "refined_success": _FakeArray(np.ones((1,), dtype=bool)),
        },
        attrs={"keypoint_labels": ["swim_bladder"]},
    )
    refined = SimpleNamespace(
        run_name="refined_001",
        component_to_index={"swim_bladder": 0},
        group=_FakeGroup(
            {
                "masks_roi": _FakeArray(np.zeros((1, 1, 8, 8), dtype=np.uint8)),
            },
            attrs={},
        ),
    )
    swim_source = SimpleNamespace(
        crop_run="crop_001",
        group=_FakeGroup(
            {
                "masks_roi": _FakeArray(np.zeros((2, 1, 8, 8), dtype=np.uint8)),
            }
        ),
        mask_labels=["swim_bladder"],
        available_channels=np.asarray([True], dtype=bool),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(swim_mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        swim_mod,
        "prepare_refined_subject_run",
        lambda *_args, **_kwargs: (SimpleNamespace(), refined),
    )
    monkeypatch.setattr(
        swim_mod,
        "_load_refined_component_source_runs",
        lambda *_args, **_kwargs: (SimpleNamespace(), {"swim_bladder": swim_source}),
    )
    monkeypatch.setattr(
        swim_mod.CropImageSource,
        "open",
        classmethod(lambda _cls, *_args, **_kwargs: crop_source),
    )
    monkeypatch.setattr(
        swim_mod,
        "_resolve_keypoint_group",
        lambda *_args, **_kwargs: (keypoint_group, "refined_keypoints_runs", "refined_kp_001"),
    )

    def _capture_alignment(actual_crop_group, crop_run_name, kp_group, *, total_rois) -> None:
        captured["crop_group"] = actual_crop_group
        captured["crop_run_name"] = crop_run_name
        captured["kp_group"] = kp_group
        captured["total_rois"] = total_rois

    monkeypatch.setattr(swim_mod, "_validate_keypoint_group_alignment", _capture_alignment)
    monkeypatch.setattr(swim_mod, "_require_gui_display", lambda: (_ for _ in ()).throw(_StopViewer()))

    with pytest.raises(_StopViewer):
        swim_mod.create_viewer(
            tmp_path / "archive.zarr",
            subject_run="subject_001",
            refined_run="refined_001",
            crop_run=None,
            keypoint_run=None,
            keypoint_group=None,
            start_roi=0,
            roi_indices=None,
            padding=18,
            scale_percent=220,
            edit_zoom=8,
            review_state="approved",
            review_method="manual",
            review_intended_use="training",
            reviewer=None,
            review_notes=None,
        )

    assert captured["crop_group"] is crop_group
    assert captured["crop_run_name"] == "crop_001"
    assert captured["kp_group"] is keypoint_group
    assert captured["total_rois"] == 1


def test_create_viewer_navigation_uses_roi_slider_as_single_source_of_truth(monkeypatch, tmp_path) -> None:
    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.total_rois = 2
            self.read_log: list[int] = []

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            self.read_log.extend(range(int(start), int(stop)))
            batch = np.zeros((max(0, int(stop) - int(start)), 8, 8), dtype=np.uint8)
            for idx, roi_idx in enumerate(range(int(start), int(stop))):
                batch[idx].fill(int(roi_idx))
            return batch

    root = _FakeGroup()
    crop_group = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.asarray([0, 1], dtype=np.int32)),
            "detection_indices": _FakeArray(np.asarray([10, 11], dtype=np.int32)),
        }
    )
    crop_source = _FakeCropSource(crop_group)
    keypoint_group = _FakeGroup(
        {
            "keypoints_roi": _FakeArray(np.asarray([[[1.0, 1.0]], [[1.0, 1.0]]], dtype=np.float32)),
            "refined_success": _FakeArray(np.ones((2,), dtype=bool)),
        },
        attrs={"keypoint_labels": ["swim_bladder"]},
    )
    refined = SimpleNamespace(
        run_name="refined_001",
        component_to_index={"swim_bladder": 0},
        group=_FakeGroup(
            {
                "masks_roi": _FakeArray(np.zeros((2, 1, 8, 8), dtype=np.uint8)),
            },
            attrs={},
        ),
        parent=_FakeGroup(),
    )
    swim_source = SimpleNamespace(
        crop_run="crop_001",
        group=_FakeGroup(
            {
                "masks_roi": _FakeArray(np.zeros((2, 1, 8, 8), dtype=np.uint8)),
            }
        ),
        mask_labels=["swim_bladder"],
        available_channels=np.asarray([True], dtype=bool),
    )
    named_windows: list[str] = []
    trackbars: dict[str, int] = {}
    trackbar_windows: dict[str, str] = {}
    pending_roi: dict[str, int | None] = {"value": None}
    roi_set_calls: list[int] = []
    roi_set_windows: list[str] = []
    stale_roi_reads = {"remaining": 0}
    key_sequence = [ord("n"), 255, 255, ord("q")]

    monkeypatch.setattr(swim_mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        swim_mod,
        "prepare_refined_subject_run",
        lambda *_args, **_kwargs: (SimpleNamespace(run_name="subject_001"), refined),
    )
    monkeypatch.setattr(
        swim_mod,
        "_load_refined_component_source_runs",
        lambda *_args, **_kwargs: (SimpleNamespace(), {"swim_bladder": swim_source}),
    )
    monkeypatch.setattr(
        swim_mod.CropImageSource,
        "open",
        classmethod(lambda _cls, *_args, **_kwargs: crop_source),
    )
    monkeypatch.setattr(
        swim_mod,
        "_resolve_keypoint_group",
        lambda *_args, **_kwargs: (keypoint_group, "refined_keypoints_runs", "refined_kp_001"),
    )
    monkeypatch.setattr(swim_mod, "_validate_keypoint_group_alignment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(swim_mod, "_require_gui_display", lambda: None)

    monkeypatch.setattr(swim_mod.cv2, "namedWindow", lambda name, *args, **kwargs: named_windows.append(str(name)))
    monkeypatch.setattr(swim_mod.cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        swim_mod.cv2,
        "createTrackbar",
        lambda name, window, value, _maximum, _cb: (
            trackbar_windows.__setitem__(str(name), str(window)),
            trackbars.__setitem__(str(name), int(value)),
        )[-1],
    )
    monkeypatch.setattr(swim_mod.cv2, "setMouseCallback", lambda *args, **kwargs: None)

    def _fake_get_trackbar_pos(name: str, _window: str) -> int:
        if name == "ROI" and stale_roi_reads["remaining"] > 0:
            stale_roi_reads["remaining"] -= 1
            return int(trackbars.get(name, 0))
        if name == "ROI" and pending_roi["value"] is not None:
            trackbars[name] = int(pending_roi["value"])
            pending_roi["value"] = None
        return int(trackbars.get(name, 0))

    def _fake_set_trackbar_pos(name: str, window: str, value: int) -> None:
        if name == "ROI":
            roi_set_calls.append(int(value))
            roi_set_windows.append(str(window))
            pending_roi["value"] = int(value)
            stale_roi_reads["remaining"] = 1
            return
        trackbars[name] = int(value)

    monkeypatch.setattr(swim_mod.cv2, "getTrackbarPos", _fake_get_trackbar_pos)
    monkeypatch.setattr(swim_mod.cv2, "setTrackbarPos", _fake_set_trackbar_pos)
    monkeypatch.setattr(swim_mod.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        swim_mod.cv2,
        "waitKey",
        lambda _delay: key_sequence.pop(0) if key_sequence else ord("q"),
    )
    monkeypatch.setattr(swim_mod.cv2, "destroyAllWindows", lambda *args, **kwargs: None)

    swim_mod.create_viewer(
        tmp_path / "archive.zarr",
        subject_run="subject_001",
        refined_run="refined_001",
        crop_run=None,
        keypoint_run=None,
        keypoint_group=None,
        start_roi=0,
        roi_indices=None,
        padding=18,
        scale_percent=220,
        edit_zoom=8,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        reviewer=None,
        review_notes=None,
    )

    assert named_windows == [swim_mod.WINDOW_NAME, swim_mod.CONTROL_WINDOW_NAME]
    assert set(trackbar_windows.values()) == {swim_mod.CONTROL_WINDOW_NAME}
    assert crop_source.read_log == [0, 1]
    assert roi_set_calls == [1]
    assert roi_set_windows == [swim_mod.CONTROL_WINDOW_NAME]


def test_create_viewer_navigation_maps_roi_subset_to_actual_rows(monkeypatch, tmp_path) -> None:
    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.total_rois = 4
            self.read_log: list[int] = []

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            self.read_log.extend(range(int(start), int(stop)))
            batch = np.zeros((max(0, int(stop) - int(start)), 8, 8), dtype=np.uint8)
            for idx, roi_idx in enumerate(range(int(start), int(stop))):
                batch[idx].fill(int(roi_idx))
            return batch

    root = _FakeGroup()
    crop_group = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.asarray([0, 1, 2, 3], dtype=np.int32)),
            "detection_indices": _FakeArray(np.asarray([10, 11, 12, 13], dtype=np.int32)),
        }
    )
    crop_source = _FakeCropSource(crop_group)
    keypoint_group = _FakeGroup(
        {
            "keypoints_roi": _FakeArray(np.asarray([[[1.0, 1.0]]] * 4, dtype=np.float32)),
            "refined_success": _FakeArray(np.ones((4,), dtype=bool)),
        },
        attrs={"keypoint_labels": ["swim_bladder"]},
    )
    refined = SimpleNamespace(
        run_name="refined_001",
        component_to_index={"swim_bladder": 0},
        group=_FakeGroup(
            {
                "masks_roi": _FakeArray(np.zeros((4, 1, 8, 8), dtype=np.uint8)),
            },
            attrs={},
        ),
        parent=_FakeGroup(),
    )
    swim_source = SimpleNamespace(
        crop_run="crop_001",
        group=_FakeGroup(
            {
                "masks_roi": _FakeArray(np.zeros((4, 1, 8, 8), dtype=np.uint8)),
            }
        ),
        mask_labels=["swim_bladder"],
        available_channels=np.asarray([True], dtype=bool),
    )
    trackbars: dict[str, int] = {}
    key_sequence = [ord("n"), ord("q")]

    monkeypatch.setattr(swim_mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        swim_mod,
        "prepare_refined_subject_run",
        lambda *_args, **_kwargs: (SimpleNamespace(run_name="subject_001"), refined),
    )
    monkeypatch.setattr(
        swim_mod,
        "_load_refined_component_source_runs",
        lambda *_args, **_kwargs: (SimpleNamespace(), {"swim_bladder": swim_source}),
    )
    monkeypatch.setattr(
        swim_mod.CropImageSource,
        "open",
        classmethod(lambda _cls, *_args, **_kwargs: crop_source),
    )
    monkeypatch.setattr(
        swim_mod,
        "_resolve_keypoint_group",
        lambda *_args, **_kwargs: (keypoint_group, "refined_keypoints_runs", "refined_kp_001"),
    )
    monkeypatch.setattr(swim_mod, "_validate_keypoint_group_alignment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(swim_mod, "_require_gui_display", lambda: None)
    monkeypatch.setattr(swim_mod.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(swim_mod.cv2, "resizeWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        swim_mod.cv2,
        "createTrackbar",
        lambda name, _window, value, _maximum, _cb: trackbars.__setitem__(name, int(value)),
    )
    monkeypatch.setattr(swim_mod.cv2, "setMouseCallback", lambda *args, **kwargs: None)
    monkeypatch.setattr(swim_mod.cv2, "getTrackbarPos", lambda name, _window: int(trackbars.get(name, 0)))
    monkeypatch.setattr(
        swim_mod.cv2,
        "setTrackbarPos",
        lambda name, _window, value: trackbars.__setitem__(name, int(value)),
    )
    monkeypatch.setattr(swim_mod.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        swim_mod.cv2,
        "waitKey",
        lambda _delay: key_sequence.pop(0) if key_sequence else ord("q"),
    )
    monkeypatch.setattr(swim_mod.cv2, "destroyAllWindows", lambda *args, **kwargs: None)

    swim_mod.create_viewer(
        tmp_path / "archive.zarr",
        subject_run="subject_001",
        refined_run="refined_001",
        crop_run=None,
        keypoint_run=None,
        keypoint_group=None,
        start_roi=1,
        roi_indices=[1, 3],
        padding=18,
        scale_percent=220,
        edit_zoom=8,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        reviewer=None,
        review_notes=None,
    )

    assert crop_source.read_log == [1, 3]


def test_create_viewer_single_roi_subset_uses_safe_trackbar_max(monkeypatch, tmp_path) -> None:
    class _FakeCropSource:
        def __init__(self, crop_group) -> None:
            self.crop_group = crop_group
            self.total_rois = 3
            self.read_log: list[int] = []

        def read_slice(self, start: int, stop: int) -> np.ndarray:
            self.read_log.extend(range(int(start), int(stop)))
            return np.zeros((max(0, int(stop) - int(start)), 8, 8), dtype=np.uint8)

    root = _FakeGroup()
    crop_group = _FakeGroup(
        {
            "frame_indices": _FakeArray(np.asarray([0, 1, 2], dtype=np.int32)),
            "detection_indices": _FakeArray(np.asarray([10, 11, 12], dtype=np.int32)),
        }
    )
    crop_source = _FakeCropSource(crop_group)
    keypoint_group = _FakeGroup(
        {
            "keypoints_roi": _FakeArray(np.asarray([[[1.0, 1.0]]] * 3, dtype=np.float32)),
            "refined_success": _FakeArray(np.ones((3,), dtype=bool)),
        },
        attrs={"keypoint_labels": ["swim_bladder"]},
    )
    refined = SimpleNamespace(
        run_name="refined_001",
        component_to_index={"swim_bladder": 0},
        group=_FakeGroup(
            {
                "masks_roi": _FakeArray(np.zeros((3, 1, 8, 8), dtype=np.uint8)),
            },
            attrs={},
        ),
        parent=_FakeGroup(),
    )
    swim_source = SimpleNamespace(
        crop_run="crop_001",
        group=_FakeGroup(
            {
                "masks_roi": _FakeArray(np.zeros((3, 1, 8, 8), dtype=np.uint8)),
            }
        ),
        mask_labels=["swim_bladder"],
        available_channels=np.asarray([True], dtype=bool),
    )
    trackbars: dict[str, int] = {}
    roi_trackbar_max: dict[str, int] = {}
    key_sequence = [ord("q")]

    monkeypatch.setattr(swim_mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        swim_mod,
        "prepare_refined_subject_run",
        lambda *_args, **_kwargs: (SimpleNamespace(run_name="subject_001"), refined),
    )
    monkeypatch.setattr(
        swim_mod,
        "_load_refined_component_source_runs",
        lambda *_args, **_kwargs: (SimpleNamespace(), {"swim_bladder": swim_source}),
    )
    monkeypatch.setattr(
        swim_mod.CropImageSource,
        "open",
        classmethod(lambda _cls, *_args, **_kwargs: crop_source),
    )
    monkeypatch.setattr(
        swim_mod,
        "_resolve_keypoint_group",
        lambda *_args, **_kwargs: (keypoint_group, "refined_keypoints_runs", "refined_kp_001"),
    )
    monkeypatch.setattr(swim_mod, "_validate_keypoint_group_alignment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(swim_mod, "_require_gui_display", lambda: None)
    monkeypatch.setattr(swim_mod.cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(swim_mod.cv2, "resizeWindow", lambda *args, **kwargs: None)

    def _create_trackbar(name: str, _window: str, value: int, maximum: int, _cb) -> None:  # type: ignore[no-untyped-def]
        if name == "ROI":
            roi_trackbar_max["value"] = int(maximum)
            if int(maximum) <= 0:
                raise AssertionError("ROI trackbar max must be > 0 for OpenCV Qt.")
        trackbars[name] = int(value)

    monkeypatch.setattr(swim_mod.cv2, "createTrackbar", _create_trackbar)
    monkeypatch.setattr(swim_mod.cv2, "setMouseCallback", lambda *args, **kwargs: None)
    monkeypatch.setattr(swim_mod.cv2, "getTrackbarPos", lambda name, _window: int(trackbars.get(name, 0)))
    monkeypatch.setattr(
        swim_mod.cv2,
        "setTrackbarPos",
        lambda name, _window, value: trackbars.__setitem__(name, int(value)),
    )
    monkeypatch.setattr(swim_mod.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        swim_mod.cv2,
        "waitKey",
        lambda _delay: key_sequence.pop(0) if key_sequence else ord("q"),
    )
    monkeypatch.setattr(swim_mod.cv2, "destroyAllWindows", lambda *args, **kwargs: None)

    swim_mod.create_viewer(
        tmp_path / "archive.zarr",
        subject_run="subject_001",
        refined_run="refined_001",
        crop_run=None,
        keypoint_run=None,
        keypoint_group=None,
        start_roi=2,
        roi_indices=[2],
        padding=18,
        scale_percent=220,
        edit_zoom=8,
        review_state="approved",
        review_method="manual",
        review_intended_use="training",
        reviewer=None,
        review_notes=None,
    )

    assert roi_trackbar_max["value"] == 1
    assert crop_source.read_log == [2]


def test_build_view_returns_edit_meta() -> None:
    roi = np.zeros((16, 16), dtype=np.uint8)
    roi[4:12, 4:12] = 80
    source_mask = np.zeros((16, 16), dtype=np.uint8)
    source_mask[7:9, 7:9] = 1
    current_mask = np.zeros((16, 16), dtype=np.uint8)
    current_mask[6:10, 6:10] = 1

    canvas, edit_meta = _build_view(
        roi,
        source_mask,
        current_mask,
        center_xy=(8.0, 8.0),
        center_source="keypoint",
        padding=4,
        edit_zoom=4,
        brush_radius=3,
        cursor_patch_xy=(2, 2),
    )

    assert canvas.ndim == 3
    assert canvas.shape[2] == 3
    assert edit_meta["patch_x0"] == 4
    assert edit_meta["patch_y0"] == 4
    assert edit_meta["patch_w"] == 9
    assert edit_meta["patch_h"] == 9
    assert edit_meta["zoom"] == 4


def test_pad_canvas_to_shape_stabilizes_viewer_extent() -> None:
    canvas = np.zeros((40, 60, 3), dtype=np.uint8)
    target_shape = _target_canvas_shape((16, 24), padding=4, edit_zoom=3)

    padded = _pad_canvas_to_shape(canvas, target_shape)

    assert padded.shape[:2] == target_shape
    assert padded.shape[2] == 3
