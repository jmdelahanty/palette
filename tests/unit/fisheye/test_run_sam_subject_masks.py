from __future__ import annotations

import numpy as np
import zarr

from fisheye.utils import run_sam_subject_masks as mod


class FakeArray:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.ndim = self._data.ndim

    def __getitem__(self, item):
        return self._data[item]


class FakeGroup(dict):
    def __init__(self):
        super().__init__()
        self.attrs: dict[str, object] = {}

    def get(self, name: str, default=None):
        return super().get(name, default)


def _fake_root(*, width: int = 100, height: int = 80) -> FakeGroup:
    root = FakeGroup()
    root.attrs["width"] = int(width)
    root.attrs["height"] = int(height)
    return root


def _xyxy_to_norm_xywh(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    frame_width: int = 100,
    frame_height: int = 80,
) -> np.ndarray:
    w = float(x1 - x0)
    h = float(y1 - y0)
    cx = float(x0) + (w / 2.0)
    cy = float(y0) + (h / 2.0)
    return np.asarray(
        [cx / float(frame_width), cy / float(frame_height), w / float(frame_width), h / float(frame_height)],
        dtype=np.float32,
    )


def _fake_crop_group(*, detection_source: np.ndarray | None = None) -> FakeGroup:
    group = FakeGroup()
    if detection_source is None:
        detection_source = np.asarray([0, 1, 0], dtype=np.int8)
    group["roi_images"] = FakeArray(np.arange(3 * 6 * 8, dtype=np.uint8).reshape(3, 6, 8))
    group["roi_coordinates_full"] = FakeArray(
        np.asarray(
            [
                [10, 20],
                [30, 40],
                [60, 10],
            ],
            dtype=np.int32,
        )
    )
    group["bbox_norm_coords"] = FakeArray(
        np.asarray(
            [
                _xyxy_to_norm_xywh(12, 21, 16, 25),
                _xyxy_to_norm_xywh(33, 41, 37, 45),
                _xyxy_to_norm_xywh(60, 10, 68, 16),
            ],
            dtype=np.float32,
        )
    )
    group["frame_indices"] = FakeArray(np.asarray([0, 1, 2], dtype=np.int32))
    group["detection_indices"] = FakeArray(np.asarray([10, 11, 12], dtype=np.int32))
    group["detection_source"] = FakeArray(np.asarray(detection_source, dtype=np.int8))
    group.attrs["crop_storage_mode"] = "materialized"
    return group


def _fake_keypoint_group(
    *,
    frame_indices: np.ndarray | None = None,
    detection_indices: np.ndarray | None = None,
    detection_source: np.ndarray | None = None,
    keypoints_roi: np.ndarray | None = None,
    refined_success: np.ndarray | None = None,
    geometry_valid: np.ndarray | None = None,
    usable_keypoints: np.ndarray | None = None,
) -> FakeGroup:
    group = FakeGroup()
    if frame_indices is None:
        frame_indices = np.asarray([0, 1, 2], dtype=np.int32)
    if detection_indices is None:
        detection_indices = np.asarray([10, 11, 12], dtype=np.int32)
    if detection_source is None:
        detection_source = np.asarray([0, 1, 0], dtype=np.int8)
    if keypoints_roi is None:
        keypoints_roi = np.asarray(
            [
                [[2.0, 2.0], [3.0, 2.0], [1.0, 2.0]],
                [[4.0, 2.0], [5.0, 2.0], [3.0, 2.0]],
                [[99.0, 99.0], [4.0, 4.0], [2.0, 4.0]],
            ],
            dtype=np.float32,
        )
    group["frame_indices"] = FakeArray(frame_indices)
    group["detection_indices"] = FakeArray(detection_indices)
    group["detection_source"] = FakeArray(detection_source)
    group["keypoints_roi"] = FakeArray(keypoints_roi)
    group.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    if refined_success is not None:
        group["refined_success"] = FakeArray(np.asarray(refined_success, dtype=bool))
    if geometry_valid is not None:
        group["geometry_valid"] = FakeArray(np.asarray(geometry_valid, dtype=bool))
    if usable_keypoints is not None:
        group["usable_keypoints"] = FakeArray(np.asarray(usable_keypoints, dtype=bool))
    return group


def test_inspect_sam_subject_archive_prepares_expected_preview(monkeypatch) -> None:
    root = _fake_root()
    crop_group = _fake_crop_group()
    keypoint_group_obj = _fake_keypoint_group(
        refined_success=np.asarray([True, True, True], dtype=bool),
        geometry_valid=np.asarray([True, True, True], dtype=bool),
        usable_keypoints=np.asarray([True, True, True], dtype=bool),
    )

    monkeypatch.setattr(
        mod,
        "resolve_materialized_crop_run",
        lambda root, *, crop_run=None: (FakeGroup(), crop_group, "crop_001"),
    )
    monkeypatch.setattr(
        mod,
        "_resolve_keypoint_run",
        lambda root, *, keypoint_run=None, keypoint_group="auto": mod.ResolvedKeypointRun(
            group_name="refined_keypoints_runs",
            run_name="refined_001",
            group=keypoint_group_obj,
        ),
    )

    summary = mod.inspect_sam_subject_archive(root, inspect_runtime=False, prepare_count=2)

    assert summary["crop_run"] == "crop_001"
    assert summary["keypoint_group"] == "refined_keypoints_runs"
    assert summary["keypoint_run"] == "refined_001"
    assert summary["keypoint_labels"] == ["swim_bladder", "eye_left", "eye_right"]
    assert summary["positive_keypoint_labels"] == ["swim_bladder", "eye_left", "eye_right"]
    assert summary["alignment"]["status"] == "ok"
    assert summary["eligibility"]["eligible_rows"] == 2
    assert summary["eligibility"]["skipped_interpolated_rows"] == 1
    assert summary["eligibility"]["off_image_prompt_rows"] == 0
    assert summary["prepared_batch"]["prepared_rows"] == 2
    assert summary["prepared_batch"]["row_indices"] == [0, 2]
    assert summary["prepared_batch"]["image_shape"] == [6, 8, 3]
    assert summary["prepared_batch"]["point_coords_shape"] == [3, 2]
    assert summary["prepared_batch"]["positive_point_count"] == 3
    assert summary["prepared_batch"]["box_shape"] == [1, 4]
    assert summary["planned_output"]["available_channels"] == [True, False, False]
    assert summary["planned_output"]["sam_prompt_policy"] == "keypoint_points_plus_detect_box"
    assert "sam3_runtime" not in summary


def test_inspect_sam_subject_archive_supports_explicit_positive_keypoint_labels(monkeypatch) -> None:
    root = _fake_root()
    crop_group = _fake_crop_group()
    keypoint_group_obj = _fake_keypoint_group(
        refined_success=np.asarray([True, True, True], dtype=bool),
        geometry_valid=np.asarray([True, True, True], dtype=bool),
        usable_keypoints=np.asarray([True, True, True], dtype=bool),
    )

    monkeypatch.setattr(
        mod,
        "resolve_materialized_crop_run",
        lambda root, *, crop_run=None: (FakeGroup(), crop_group, "crop_001"),
    )
    monkeypatch.setattr(
        mod,
        "_resolve_keypoint_run",
        lambda root, *, keypoint_run=None, keypoint_group="auto": mod.ResolvedKeypointRun(
            group_name="refined_keypoints_runs",
            run_name="refined_001",
            group=keypoint_group_obj,
        ),
    )

    summary = mod.inspect_sam_subject_archive(
        root,
        inspect_runtime=False,
        prepare_count=2,
        positive_keypoint_labels=["swim_bladder"],
    )

    assert summary["positive_keypoint_labels"] == ["swim_bladder"]
    assert summary["eligibility"]["eligible_rows"] == 1
    assert summary["eligibility"]["off_image_prompt_rows"] == 1
    assert summary["prepared_batch"]["point_coords_shape"] == [1, 2]


def test_resolve_keypoint_run_prefers_refined_keypoints(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def _fake_resolve(root, parent_path, run_name, **kwargs):
        calls.append((str(parent_path), run_name))
        if str(parent_path) == "refined_keypoints_runs":
            return FakeGroup(), "refined_001"
        raise AssertionError("keypoints_runs should not be consulted after refined succeeds")

    monkeypatch.setattr(mod, "resolve_zarr_run", _fake_resolve)

    resolved = mod._resolve_keypoint_run(FakeGroup(), keypoint_run=None, keypoint_group="auto")

    assert resolved.group_name == "refined_keypoints_runs"
    assert resolved.run_name == "refined_001"
    assert calls == [("refined_keypoints_runs", None)]


def test_resolve_keypoint_run_falls_back_to_raw_keypoints(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_resolve(root, parent_path, run_name, **kwargs):
        calls.append(str(parent_path))
        if str(parent_path) == "refined_keypoints_runs":
            raise ValueError("no refined run")
        return FakeGroup(), "raw_001"

    monkeypatch.setattr(mod, "resolve_zarr_run", _fake_resolve)

    resolved = mod._resolve_keypoint_run(FakeGroup(), keypoint_run=None, keypoint_group="auto")

    assert resolved.group_name == "keypoints_runs"
    assert resolved.run_name == "raw_001"
    assert calls == ["refined_keypoints_runs", "keypoints_runs"]


def test_resolve_sam_subject_inputs_raises_on_alignment_mismatch(monkeypatch) -> None:
    root = _fake_root()
    crop_group = _fake_crop_group()
    keypoint_group_obj = _fake_keypoint_group(frame_indices=np.asarray([0, 99, 2], dtype=np.int32))

    monkeypatch.setattr(
        mod,
        "resolve_materialized_crop_run",
        lambda root, *, crop_run=None: (FakeGroup(), crop_group, "crop_001"),
    )
    monkeypatch.setattr(
        mod,
        "_resolve_keypoint_run",
        lambda root, *, keypoint_run=None, keypoint_group="auto": mod.ResolvedKeypointRun(
            group_name="refined_keypoints_runs",
            run_name="refined_001",
            group=keypoint_group_obj,
        ),
    )

    try:
        mod.resolve_sam_subject_inputs(root)
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError for frame_indices mismatch")

    assert "frame_indices mismatch at row 1" in message


def test_convert_roi_batch_to_rgb_repeats_single_channel_and_scales_float01() -> None:
    grayscale = np.asarray(
        [
            [[[0.0], [1.0]]],
            [[[0.5], [0.25]]],
        ],
        dtype=np.float32,
    )

    rgb = mod.convert_roi_batch_to_rgb(grayscale)

    assert rgb.shape == (2, 1, 2, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0, 0], np.asarray([0, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(rgb[0, 0, 1], np.asarray([255, 255, 255], dtype=np.uint8))
    np.testing.assert_array_equal(rgb[1, 0, 0], np.asarray([127, 127, 127], dtype=np.uint8))


def test_build_sam_batch_for_rows_preserves_requested_order() -> None:
    inputs = mod.ResolvedSamInputs(
        crop_run="crop_001",
        keypoint_group="refined_keypoints_runs",
        keypoint_run="refined_001",
        roi_images=FakeArray(np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)),
        roi_coordinates_full=np.asarray([[10, 20], [30, 40], [50, 60]], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                _xyxy_to_norm_xywh(11, 21, 14, 24),
                _xyxy_to_norm_xywh(31, 40, 34, 43),
                _xyxy_to_norm_xywh(52, 61, 54, 63),
            ],
            dtype=np.float32,
        ),
        frame_indices=np.asarray([0, 1, 2], dtype=np.int32),
        detection_indices=np.asarray([10, 11, 12], dtype=np.int32),
        detection_source=np.asarray([0, 0, 0], dtype=np.int8),
        keypoints_roi=np.asarray(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[2.0, 3.0], [2.0, 2.0], [3.0, 3.0]],
            ],
            dtype=np.float32,
        ),
        keypoint_labels=("swim_bladder", "eye_left", "eye_right"),
        success_flags=None,
        geometry_valid=None,
        usable_keypoints=None,
        frame_height=80,
        frame_width=100,
        warnings=(),
    )

    selection = mod.resolve_prompt_keypoint_selection(inputs)
    batch = mod.build_sam_batch_for_rows(
        inputs,
        np.asarray([2, 0], dtype=np.int32),
        prompt_selection=selection,
    )

    assert batch.row_indices.tolist() == [2, 0]
    assert len(batch.images_rgb) == 2
    assert batch.images_rgb[0].shape == (4, 5, 3)
    np.testing.assert_array_equal(
        batch.point_coords_batch[0],
        np.asarray([[2.0, 3.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch.point_coords_batch[1],
        np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        batch.box_batch[0],
        np.asarray([[2.0, 1.0, 4.0, 3.0]], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        batch.box_batch[1],
        np.asarray([[1.0, 1.0, 4.0, 3.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_build_sam_batch_for_rows_supports_roi_inset_box_prompt() -> None:
    inputs = mod.ResolvedSamInputs(
        crop_run="crop_001",
        keypoint_group="refined_keypoints_runs",
        keypoint_run="refined_001",
        roi_images=FakeArray(np.arange(2 * 20 * 30, dtype=np.uint8).reshape(2, 20, 30)),
        roi_coordinates_full=np.asarray([[10, 20], [30, 40]], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                _xyxy_to_norm_xywh(11, 21, 14, 24),
                _xyxy_to_norm_xywh(31, 40, 34, 43),
            ],
            dtype=np.float32,
        ),
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        detection_indices=np.asarray([10, 11], dtype=np.int32),
        detection_source=np.asarray([0, 0], dtype=np.int8),
        keypoints_roi=np.asarray(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            ],
            dtype=np.float32,
        ),
        keypoint_labels=("swim_bladder", "eye_left", "eye_right"),
        success_flags=None,
        geometry_valid=None,
        usable_keypoints=None,
        frame_height=80,
        frame_width=100,
        warnings=(),
    )

    batch = mod.build_sam_batch_for_rows(
        inputs,
        np.asarray([1, 0], dtype=np.int32),
        prompt_selection=mod.resolve_prompt_keypoint_selection(inputs),
        box_prompt_source="roi_inset",
        roi_inset_fraction=0.10,
    )

    assert batch.row_indices.tolist() == [1, 0]
    assert batch.box_batch is not None
    np.testing.assert_allclose(
        batch.box_batch[0],
        np.asarray([[3.0, 2.0, 26.0, 17.0]], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        batch.box_batch[1],
        np.asarray([[3.0, 2.0, 26.0, 17.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_build_point_prompt_coords_labels_adds_corner_negatives() -> None:
    coords, labels = mod.build_point_prompt_coords_labels(
        np.asarray([11.0, 7.0], dtype=np.float32),
        roi_height=20,
        roi_width=30,
        negative_point_policy="corners",
        negative_point_margin_fraction=0.10,
    )

    np.testing.assert_allclose(
        coords,
        np.asarray(
            [
                [11.0, 7.0],
                [3.0, 2.0],
                [26.0, 2.0],
                [3.0, 17.0],
                [26.0, 17.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(labels, np.asarray([1, 0, 0, 0, 0], dtype=np.int32))


def test_build_sam_batch_for_rows_supports_negative_point_policy() -> None:
    inputs = mod.ResolvedSamInputs(
        crop_run="crop_001",
        keypoint_group="refined_keypoints_runs",
        keypoint_run="refined_001",
        roi_images=FakeArray(np.arange(2 * 20 * 30, dtype=np.uint8).reshape(2, 20, 30)),
        roi_coordinates_full=np.asarray([[10, 20], [30, 40]], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                _xyxy_to_norm_xywh(11, 21, 14, 24),
                _xyxy_to_norm_xywh(31, 40, 34, 43),
            ],
            dtype=np.float32,
        ),
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        detection_indices=np.asarray([10, 11], dtype=np.int32),
        detection_source=np.asarray([0, 0], dtype=np.int8),
        keypoints_roi=np.asarray(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                [[4.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            ],
            dtype=np.float32,
        ),
        keypoint_labels=("swim_bladder", "eye_left", "eye_right"),
        success_flags=None,
        geometry_valid=None,
        usable_keypoints=None,
        frame_height=80,
        frame_width=100,
        warnings=(),
    )

    batch = mod.build_sam_batch_for_rows(
        inputs,
        np.asarray([1], dtype=np.int32),
        prompt_selection=mod.resolve_prompt_keypoint_selection(inputs, positive_keypoint_labels=["swim_bladder"]),
        negative_point_policy="corners",
        negative_point_margin_fraction=0.10,
    )

    assert batch.box_batch is not None
    np.testing.assert_allclose(
        batch.point_coords_batch[0],
        np.asarray(
            [
                [4.0, 1.0],
                [3.0, 2.0],
                [26.0, 2.0],
                [3.0, 17.0],
                [26.0, 17.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(batch.point_labels_batch[0], np.asarray([1, 0, 0, 0, 0], dtype=np.int32))


def test_project_bbox_norm_to_roi_xyxy_clips_to_roi_bounds() -> None:
    box = mod.project_bbox_norm_to_roi_xyxy(
        _xyxy_to_norm_xywh(8, 18, 20, 30),
        np.asarray([10, 20], dtype=np.int32),
        frame_height=80,
        frame_width=100,
        roi_height=6,
        roi_width=8,
    )

    np.testing.assert_allclose(box, np.asarray([0.0, 0.0, 7.0, 5.0], dtype=np.float32))


def test_build_roi_inset_box_xyxy_insets_from_roi_edges() -> None:
    box = mod.build_roi_inset_box_xyxy(roi_height=20, roi_width=30, inset_fraction=0.10)

    np.testing.assert_allclose(box, np.asarray([3.0, 2.0, 26.0, 17.0], dtype=np.float32))


def test_build_negative_prompt_points_supports_border8_policy() -> None:
    points = mod.build_negative_prompt_points(
        roi_height=20,
        roi_width=30,
        negative_point_policy="border8",
        margin_fraction=0.10,
    )

    np.testing.assert_allclose(
        points,
        np.asarray(
            [
                [3.0, 2.0],
                [26.0, 2.0],
                [3.0, 17.0],
                [26.0, 17.0],
                [14.0, 2.0],
                [14.0, 17.0],
                [3.0, 9.0],
                [26.0, 9.0],
            ],
            dtype=np.float32,
        ),
    )


def test_select_best_masks_chooses_highest_iou_candidate() -> None:
    selected = mod._select_best_masks(
        np.asarray([5, 6], dtype=np.int32),
        masks_list=[
            np.asarray(
                [
                    [[-1.0, -1.0], [-1.0, -1.0]],
                    [[2.0, -2.0], [-2.0, -2.0]],
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    [[-3.0, -3.0], [-3.0, -3.0]],
                    [[-2.0, 2.0], [-2.0, -2.0]],
                ],
                dtype=np.float32,
            ),
        ],
        ious_list=[
            np.asarray([0.1, 0.9], dtype=np.float32),
            np.asarray([0.2, 0.8], dtype=np.float32),
        ],
    )

    assert selected.row_indices.tolist() == [5, 6]
    np.testing.assert_array_equal(
        selected.binary,
        np.asarray(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            dtype=np.uint8,
        ),
    )
    np.testing.assert_allclose(selected.scores, np.asarray([0.9, 0.8], dtype=np.float32))
    assert float(selected.probs[0, 0, 0]) > 0.5
    assert float(selected.probs[0, 0, 1]) < 0.5


def test_compute_channel_metrics_reports_area_centroid_and_bbox() -> None:
    binary = np.asarray(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    probs = binary.astype(np.float32)

    metrics = mod._compute_channel_metrics(binary, probs)

    np.testing.assert_array_equal(metrics["mask_present"], np.asarray([True, False], dtype=bool))
    np.testing.assert_array_equal(metrics["area_px"], np.asarray([2.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(metrics["centroid_xy"][0], np.asarray([0.5, 0.5], dtype=np.float32))
    np.testing.assert_array_equal(metrics["centroid_valid"], np.asarray([True, False], dtype=bool))
    np.testing.assert_array_equal(
        metrics["bbox_xyxy"][0],
        np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(metrics["bbox_valid"], np.asarray([True, False], dtype=bool))


def test_write_sam_subject_mask_run_records_richer_stage_provenance(monkeypatch) -> None:
    root = zarr.group()
    crop_parent = root.create_group("crop_runs")
    crop_group = crop_parent.create_group("crop_001")
    crop_group.attrs["video_source_type"] = "video"
    crop_group.attrs["video_source_path"] = "/tmp/source.mp4"
    crop_group.create_array("frame_indices", data=np.asarray([0, 1], dtype=np.int32))
    crop_group.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    crop_group.create_array("detection_indices", data=np.asarray([10, 11], dtype=np.int32))
    crop_group.create_array("detection_source", data=np.asarray([0, 0], dtype=np.int8))

    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "a" * 40,
            "short_hash": "aaaaaaaa",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.12"},
            "platform": {
                "hostname": "test-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.12.0",
                "machine": "x86_64",
            },
        },
    )
    monkeypatch.setattr(mod.sys, "argv", ["scripts/py", "-m", "fisheye.utils.run_sam_subject_masks", "--apply"])
    monkeypatch.setattr(mod, "_utc_now", lambda: "2026-03-01T00:00:00+00:00")

    inputs = mod.ResolvedSamInputs(
        crop_run="crop_001",
        keypoint_group="refined_keypoints_runs",
        keypoint_run="refined_001",
        roi_images=np.zeros((2, 6, 8), dtype=np.uint8),
        roi_coordinates_full=np.asarray([[10, 20], [30, 40]], dtype=np.int32),
        bbox_norm_coords=np.asarray(
            [
                _xyxy_to_norm_xywh(12, 21, 16, 25),
                _xyxy_to_norm_xywh(31, 41, 36, 45),
            ],
            dtype=np.float32,
        ),
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        detection_indices=np.asarray([10, 11], dtype=np.int32),
        detection_source=np.asarray([0, 0], dtype=np.int8),
        keypoints_roi=np.asarray(
            [
                [[2.0, 2.0], [3.0, 2.0], [1.0, 2.0]],
                [[4.0, 2.0], [5.0, 2.0], [3.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        keypoint_labels=("swim_bladder", "eye_left", "eye_right"),
        success_flags=np.asarray([True, True], dtype=bool),
        geometry_valid=np.asarray([True, True], dtype=bool),
        usable_keypoints=np.asarray([True, True], dtype=bool),
        frame_height=80,
        frame_width=100,
        warnings=(),
    )
    eligibility = mod.RowEligibility(
        eligible=np.asarray([True, False], dtype=bool),
        prompt_point_finite=np.asarray([True, True], dtype=bool),
        prompt_point_in_bounds=np.asarray([True, True], dtype=bool),
        prompt_point_count=np.asarray([3, 3], dtype=np.int32),
        skipped_interpolated=np.asarray([False, False], dtype=bool),
        success_ok=np.asarray([True, True], dtype=bool),
        geometry_ok=np.asarray([True, True], dtype=bool),
        usable_ok=np.asarray([True, True], dtype=bool),
    )
    selected = mod.SelectedMaskBatch(
        row_indices=np.asarray([0], dtype=np.int32),
        binary=np.ones((1, 6, 8), dtype=np.uint8),
        probs=np.full((1, 6, 8), 0.75, dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
    )
    prompt_selection = mod.PromptKeypointSelection(
        labels=("swim_bladder", "eye_left", "eye_right"),
        indices=np.asarray([0, 1, 2], dtype=np.int32),
    )

    summary = mod.write_sam_subject_mask_run(
        root,
        zarr_path="/tmp/fake_training.zarr",
        inputs=inputs,
        eligibility=eligibility,
        selected=selected,
        crop_group=crop_group,
        prompt_selection=prompt_selection,
        output_run="sam_subject_masks_test_001",
        overwrite=False,
        checkpoint_path="/tmp/sam3.pt",
        sam3_root="/tmp/sam3",
        multimask_output=True,
        use_box_prompt=True,
        box_prompt_source="detect",
        roi_inset_fraction=0.05,
        negative_point_policy="none",
        negative_point_margin_fraction=0.05,
        device="cpu",
        duration_seconds=1.5,
    )

    assert summary["checkpoint_path"] == "/tmp/sam3.pt"

    run = root["subject_mask_runs"]["sam_subject_masks_test_001"]
    assert run.attrs["source_keypoints_run"] == "refined_001"
    assert run.attrs["source_keypoint_run"] == "refined_001"
    assert run.attrs["git_commit"] == "a" * 40
    assert run.attrs["git_branch"] == "main"
    assert tuple(run["masks_roi"].chunks) == (2, 1, 6, 8)
    assert run["masks_roi"].fill_value == 0
    assert tuple(run["mask_probs_roi"].chunks) == (2, 1, 6, 8)
    assert run["mask_probs_roi"].fill_value == np.float16(0.0)

    provenance = run.attrs["provenance"]
    assert provenance["stage"] == "subject_masks"
    assert provenance["command"] == "scripts/py -m fisheye.utils.run_sam_subject_masks --apply"
    assert provenance["git"]["commit"] == "a" * 40
    assert provenance["environment"] == {"python": "3.12"}
    assert provenance["platform"]["hostname"] == "test-host"
    assert provenance["parameters"]["run_semantics"] == "sam_body_mask_inference"
    assert provenance["parameters"]["input_format"] == "gray"
    assert provenance["inputs"]["source_keypoints_run"] == "refined_001"
    assert provenance["inputs"]["source_keypoint_group"] == "refined_keypoints_runs"
    assert provenance["inputs"]["source_video_path"] == "/tmp/source.mp4"
    assert "source_keypoint_run" not in provenance["inputs"]


def test_run_sam_subject_mask_inference_uses_pixel_prompt_normalization(monkeypatch) -> None:
    root = _fake_root()
    crop_group = _fake_crop_group(detection_source=np.asarray([0], dtype=np.int8))
    inputs = mod.ResolvedSamInputs(
        crop_run="crop_001",
        keypoint_group="refined_keypoints_runs",
        keypoint_run="refined_001",
        roi_images=FakeArray(np.arange(1 * 6 * 8, dtype=np.uint8).reshape(1, 6, 8)),
        roi_coordinates_full=np.asarray([[10, 20]], dtype=np.int32),
        bbox_norm_coords=np.asarray([_xyxy_to_norm_xywh(12, 21, 16, 25)], dtype=np.float32),
        frame_indices=np.asarray([0], dtype=np.int32),
        detection_indices=np.asarray([10], dtype=np.int32),
        detection_source=np.asarray([0], dtype=np.int8),
        keypoints_roi=np.asarray([[[2.0, 3.0], [3.0, 3.0], [1.0, 3.0]]], dtype=np.float32),
        keypoint_labels=("swim_bladder", "eye_left", "eye_right"),
        success_flags=np.asarray([True], dtype=bool),
        geometry_valid=np.asarray([True], dtype=bool),
        usable_keypoints=np.asarray([True], dtype=bool),
        frame_height=80,
        frame_width=100,
        warnings=(),
    )
    eligibility = mod.RowEligibility(
        eligible=np.asarray([True], dtype=bool),
        prompt_point_finite=np.asarray([True], dtype=bool),
        prompt_point_in_bounds=np.asarray([True], dtype=bool),
        prompt_point_count=np.asarray([3], dtype=np.int32),
        skipped_interpolated=np.asarray([False], dtype=bool),
        success_ok=np.asarray([True], dtype=bool),
        geometry_ok=np.asarray([True], dtype=bool),
        usable_ok=np.asarray([True], dtype=bool),
    )

    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)
    monkeypatch.setattr(
        mod,
        "resolve_materialized_crop_run",
        lambda *_args, **_kwargs: (mod.PreparedSamBatch, crop_group, "crop_001"),
    )
    monkeypatch.setattr(mod, "resolve_sam_subject_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(mod, "compute_row_eligibility", lambda *_args, **_kwargs: eligibility)

    captured: dict[str, object] = {}

    class _FakeModel:
        def predict_inst_batch(self, inference_state, point_coords_batch, point_labels_batch, **kwargs):
            captured["inference_state"] = inference_state
            captured["point_coords_batch"] = point_coords_batch
            captured["point_labels_batch"] = point_labels_batch
            captured.update(kwargs)
            return (
                [np.asarray([[[1.0, -1.0], [-1.0, -1.0]]], dtype=np.float32)],
                [np.asarray([0.9], dtype=np.float32)],
                [np.asarray([[[1.0, -1.0], [-1.0, -1.0]]], dtype=np.float32)],
            )

    class _FakeProcessor:
        def __init__(self, model):
            self.model = model

        def set_image_batch(self, images):
            captured["processor_images"] = images
            return {"fake": True}

    monkeypatch.setattr(
        mod,
        "_load_sam3_builder",
        lambda _sam3_root: (
            "/tmp/sam3",
            lambda **_kwargs: _FakeModel(),
            _FakeProcessor,
            type("_FakePilImage", (), {"fromarray": staticmethod(lambda image: image)}),
            False,
        ),
    )
    monkeypatch.setattr(mod, "_resolve_runtime_device", lambda *_args, **_kwargs: "cpu")
    monkeypatch.setattr(
        mod,
        "write_sam_subject_mask_run",
        lambda *_args, **_kwargs: {"rows_segmented": 1, "rows_with_nonempty_masks": 1},
    )

    result = mod.run_sam_subject_mask_inference("/tmp/fake_training.zarr", batch_size=1)

    assert result["rows_segmented"] == 1
    assert result["positive_keypoint_labels"] == ["swim_bladder", "eye_left", "eye_right"]
    assert captured["normalize_coords"] is True
    assert captured["multimask_output"] is True
    np.testing.assert_array_equal(
        captured["point_coords_batch"][0],
        np.asarray([[2.0, 3.0], [3.0, 3.0], [1.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        captured["point_labels_batch"][0],
        np.asarray([1, 1, 1], dtype=np.int32),
    )


def test_resolve_prompt_keypoint_selection_supports_aliases_and_default_all() -> None:
    inputs = mod.ResolvedSamInputs(
        crop_run="crop_001",
        keypoint_group="refined_keypoints_runs",
        keypoint_run="refined_001",
        roi_images=FakeArray(np.arange(1 * 4 * 5, dtype=np.uint8).reshape(1, 4, 5)),
        roi_coordinates_full=np.asarray([[10, 20]], dtype=np.int32),
        bbox_norm_coords=np.asarray([_xyxy_to_norm_xywh(11, 21, 14, 24)], dtype=np.float32),
        frame_indices=np.asarray([0], dtype=np.int32),
        detection_indices=np.asarray([10], dtype=np.int32),
        detection_source=np.asarray([0], dtype=np.int8),
        keypoints_roi=np.asarray([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]], dtype=np.float32),
        keypoint_labels=("swim_bladder", "eye_left", "eye_right"),
        success_flags=None,
        geometry_valid=None,
        usable_keypoints=None,
        frame_height=80,
        frame_width=100,
        warnings=(),
    )

    default_selection = mod.resolve_prompt_keypoint_selection(inputs)
    alias_selection = mod.resolve_prompt_keypoint_selection(
        inputs,
        positive_keypoint_labels=["bladder", "left_eye"],
    )

    assert default_selection.labels == ("swim_bladder", "eye_left", "eye_right")
    np.testing.assert_array_equal(default_selection.indices, np.asarray([0, 1, 2], dtype=np.int32))
    assert alias_selection.labels == ("swim_bladder", "eye_left")
    np.testing.assert_array_equal(alias_selection.indices, np.asarray([0, 1], dtype=np.int32))
