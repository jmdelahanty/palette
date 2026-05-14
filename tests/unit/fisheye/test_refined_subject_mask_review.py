from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import zarr

from fisheye.shared.detect_reason_codec import read_reason_labels
from fisheye.shared.subject_mask_chunks import (
    refined_subject_mask_metric_row_chunk,
    refined_subject_mask_storage_chunks,
)
from fisheye.tune import refined_subject_mask_review as mod


def _build_subject_review_root():
    root = zarr.group()

    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "geometry_only"
    crop.attrs["crop_signature"] = {"signature_version": 2, "crop_revision": 4}
    crop.attrs["crop_revision"] = 4
    crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_001/review_status"
    roi_images = np.zeros((2, 8, 8), dtype=np.uint8)
    roi_images[0, 1:7, 1:7] = 70
    roi_images[1, 2:6, 2:6] = 120
    crop.create_array("roi_images", data=roi_images)

    subject_parent = root.create_group("subject_mask_runs")
    subject_parent.attrs["latest"] = "subject_masks_001"
    subject = subject_parent.create_group("subject_masks_001")
    subject.attrs["source_crop_run"] = "crop_001"
    subject.attrs["source_crop_storage_mode"] = "geometry_only"
    subject.attrs["source_crop_signature"] = "{'signature_version': 2, 'crop_revision': 4}"
    subject.attrs["source_crop_revision"] = 4
    subject.attrs["source_detect_review_status_ref"] = "refined_detect_runs/refined_detect_001/review_status"
    subject.attrs["method"] = "subject_mask_threshold_lr_v1"
    subject.attrs["mask_labels"] = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    subject.attrs["label_schema_id"] = "subject_v1_lr"
    subject.attrs["source_keypoint_group"] = "refined_keypoints_runs"
    subject.attrs["source_keypoints_run"] = "refined_kp_001"
    subject.create_array("detection_source", data=np.zeros((2,), dtype=np.int8))
    subject.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32))
    subject.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32))
    subject.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32))
    subject.create_array("available_channels", data=np.asarray([True, True, True, False], dtype=np.bool_))

    masks = np.zeros((2, 4, 8, 8), dtype=np.uint8)
    masks[0, 0, 1:7, 1:7] = 1
    masks[1, 0, 2:6, 2:6] = 1
    masks[0, 1, 2:3, 2:3] = 1
    masks[0, 2, 2:3, 4:5] = 1
    subject.create_array("masks_roi", data=masks)

    metrics = subject.create_group("metrics")
    metrics.create_array(
        "mask_present",
        data=np.asarray(
            [
                [True, True, True, False],
                [True, False, False, False],
            ],
            dtype=np.bool_,
        ),
    )

    return root


def _patch_review_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "b" * 40,
            "short_hash": "bbbbbbbb",
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
                "hostname": "review-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.12.0",
                "machine": "x86_64",
            },
        },
    )
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["scripts/py", "-m", "fisheye.tune.refined_subject_mask_review"],
    )


class _MinimalGroup:
    def __init__(self, attrs: dict[str, object]) -> None:
        self.attrs = attrs

    def get(self, _name: str):
        return None


def test_map_review_pointer_to_roi_maps_zoomed_edit_panel_coordinates() -> None:
    panel_regions = {
        "all": mod.ReviewPanelRegion(name="all", x=0, y=72, width=8, height=8, zoom=1),
        "edit": mod.ReviewPanelRegion(name="edit", x=16, y=72, width=32, height=32, zoom=4),
    }

    mapped = mod._map_review_pointer_to_roi(  # noqa: SLF001
        x=80,
        y=160,
        display_scale=2.0,
        panel_regions=panel_regions,
        roi_shape=(8, 8),
    )

    assert mapped == ("edit", 6, 2)


def test_map_review_pointer_to_roi_ignores_gap_and_out_of_bounds_points() -> None:
    panel_regions = {
        "all": mod.ReviewPanelRegion(name="all", x=0, y=72, width=8, height=8, zoom=1),
        "edit": mod.ReviewPanelRegion(name="edit", x=16, y=72, width=32, height=32, zoom=4),
    }

    gap_hit = mod._map_review_pointer_to_roi(  # noqa: SLF001
        x=12,
        y=74,
        display_scale=1.0,
        panel_regions=panel_regions,
        roi_shape=(8, 8),
    )
    outside_zoomed_bounds = mod._map_review_pointer_to_roi(  # noqa: SLF001
        x=16 + 33,
        y=72 + 5,
        display_scale=1.0,
        panel_regions=panel_regions,
        roi_shape=(8, 8),
    )

    assert gap_hit is None
    assert outside_zoomed_bounds is None


def test_map_review_pointer_to_roi_maps_cropped_edit_panel_with_label_and_patch_offset() -> None:
    panel_regions = {
        "edit": mod.ReviewPanelRegion(
            name="edit",
            x=40,
            y=100,
            width=24,
            height=34,
            zoom=2,
            label_h=10,
            patch_x0=7,
            patch_y0=11,
            patch_w=12,
            patch_h=12,
        ),
    }

    mapped = mod._map_review_pointer_to_roi(  # noqa: SLF001
        x=52,
        y=118,
        display_scale=1.0,
        panel_regions=panel_regions,
        roi_shape=(40, 40),
    )

    assert mapped == ("edit", 13, 15)


def test_resolve_active_patch_bounds_uses_union_of_active_and_source_masks() -> None:
    active_mask = np.zeros((20, 20), dtype=np.uint8)
    source_mask = np.zeros((20, 20), dtype=np.uint8)
    active_mask[8:10, 8:10] = 1
    source_mask[4:6, 12:14] = 1

    bounds = mod._resolve_active_patch_bounds(  # noqa: SLF001
        (20, 20),
        active_mask,
        source_mask,
        padding=2,
    )

    assert bounds == (6, 16, 2, 12)


def test_shift_patch_bounds_preserves_size_and_applies_pan() -> None:
    shifted = mod._shift_patch_bounds(  # noqa: SLF001
        (6, 16, 2, 12),
        (20, 20),
        dx=3,
        dy=-1,
    )

    assert shifted == (9, 19, 1, 11)


def test_shift_patch_bounds_clamps_to_roi_edges() -> None:
    shifted = mod._shift_patch_bounds(  # noqa: SLF001
        (6, 16, 2, 12),
        (20, 20),
        dx=10,
        dy=20,
    )

    assert shifted == (10, 20, 10, 20)


def test_fill_polygon_mask_paints_polygon_interior() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)

    ok = mod._fill_polygon_mask(  # noqa: SLF001
        mask,
        [(2, 2), (7, 2), (7, 7), (2, 7)],
        fill_value=1,
    )

    assert ok is True
    assert int(mask[4, 4]) == 1
    assert int(mask[0, 0]) == 0


def test_fill_polygon_mask_can_erase_polygon_interior() -> None:
    mask = np.ones((10, 10), dtype=np.uint8)

    ok = mod._fill_polygon_mask(  # noqa: SLF001
        mask,
        [(2, 2), (7, 2), (7, 7), (2, 7)],
        fill_value=0,
    )

    assert ok is True
    assert int(mask[4, 4]) == 0
    assert int(mask[0, 0]) == 1


def test_fill_polygon_mask_can_paint_polygon_exterior() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)

    ok = mod._fill_polygon_mask(  # noqa: SLF001
        mask,
        [(2, 2), (7, 2), (7, 7), (2, 7)],
        fill_value=1,
        invert=True,
    )

    assert ok is True
    assert int(mask[4, 4]) == 0
    assert int(mask[0, 0]) == 1


def test_fill_polygon_mask_can_erase_polygon_exterior() -> None:
    mask = np.ones((10, 10), dtype=np.uint8)

    ok = mod._fill_polygon_mask(  # noqa: SLF001
        mask,
        [(2, 2), (7, 2), (7, 7), (2, 7)],
        fill_value=0,
        invert=True,
    )

    assert ok is True
    assert int(mask[4, 4]) == 1
    assert int(mask[0, 0]) == 0


def test_append_lasso_point_skips_nearby_duplicates() -> None:
    points: list[tuple[int, int]] = []

    assert mod._append_lasso_point(points, (5, 5), min_step_px=2) is True  # noqa: SLF001
    assert mod._append_lasso_point(points, (6, 5), min_step_px=2) is False  # noqa: SLF001
    assert mod._append_lasso_point(points, (8, 5), min_step_px=2) is True  # noqa: SLF001
    assert points == [(5, 5), (8, 5)]


def test_draw_panel_cursor_crosshair_marks_center_axes() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    region = mod.ReviewPanelRegion(
        name="edit",
        x=10,
        y=8,
        width=20,
        height=22,
        zoom=2,
        label_h=4,
        patch_x0=3,
        patch_y0=5,
        patch_w=10,
        patch_h=10,
    )

    mod._draw_panel_cursor(  # noqa: SLF001
        image,
        panel_region=region,
        roi_xy=(6, 8),
        brush_radius=3,
        color=(0, 0, 255),
        style="crosshair",
    )

    center_x = 10 + round((6 - 3) * 2)
    center_y = 8 + 4 + round((8 - 5) * 2)
    assert int(image[center_y, center_x, 2]) > 0
    assert int(image[center_y, center_x - 2, 2]) > 0
    assert int(image[center_y - 2, center_x, 2]) > 0


def test_has_component_source_sync_tracking_requires_full_schema(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)

    _source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body",),
    )

    body_group = refined.group["components/subject_body"]
    assert mod._has_component_source_sync_tracking(body_group, total_rois=2) is True  # noqa: SLF001

    del body_group["manual_override"]
    body_group.attrs.pop("source_sync_schema_id", None)

    assert mod._has_component_source_sync_tracking(body_group, total_rois=2) is False  # noqa: SLF001


def test_resolve_erase_mode_uses_shift_as_temporary_inverse() -> None:
    assert mod._resolve_erase_mode(False, False) is False  # noqa: SLF001
    assert mod._resolve_erase_mode(False, True) is True  # noqa: SLF001
    assert mod._resolve_erase_mode(True, False) is True  # noqa: SLF001
    assert mod._resolve_erase_mode(True, True) is False  # noqa: SLF001


def test_compute_single_mask_topology_metrics_handles_components_and_holes() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[4:6, 4:6] = 0
    mask[0, 0] = 1

    component_count, largest_component_fraction, hole_count, hole_area_fraction = mod._compute_single_mask_topology_metrics(mask)

    assert component_count == 2
    np.testing.assert_allclose(largest_component_fraction, np.float32(32.0 / 33.0))
    assert hole_count == 1
    np.testing.assert_allclose(hole_area_fraction, np.float32(4.0 / 33.0))


def test_source_component_provenance_payload_includes_crop_snapshot_fields() -> None:
    source = mod.SourceSubjectMaskRun(
        run_name="subject_masks_001",
        group=_MinimalGroup(
            {
                "label_schema_id": "subject_v1_lr",
                "created_at_utc": "2026-04-01T00:00:00+00:00",
                "method": "subject_mask_threshold_lr_v1",
            }
        ),
        crop_run="crop_001",
        source_crop_snapshot={
            "source_crop_storage_mode": "geometry_only",
            "source_crop_signature": "{'signature_version': 2, 'crop_revision': 4}",
            "source_crop_revision": 4,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
        },
        masks_roi=np.zeros((1, 1, 1, 1), dtype=np.uint8),
        detection_source=np.zeros((1,), dtype=np.int8),
        mask_labels=("subject_body",),
        available_channels=np.asarray([True], dtype=bool),
        frame_indices=None,
        frame_counts=None,
        detection_indices=None,
        source_method="subject_mask_threshold_lr_v1",
        source_keypoints_run=None,
        source_keypoint_group=None,
    )

    payload = mod._source_component_provenance_payload(source, "subject_body")  # noqa: SLF001

    assert payload["source_stage"] == "subject_mask_runs"
    assert payload["source_run"] == "subject_masks_001"
    assert payload["source_method"] == "subject_mask_threshold_lr_v1"
    assert payload["source_crop_run"] == "crop_001"
    assert payload["source_crop_storage_mode"] == "geometry_only"
    assert payload["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert payload["source_crop_revision"] == 4
    assert payload["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"


def test_compute_single_mask_sigma_noise_is_higher_for_jagged_mask() -> None:
    clean = np.zeros((32, 32), dtype=np.uint8)
    clean[8:24, 8:24] = 1

    jagged = np.zeros((32, 32), dtype=np.uint8)
    for row in range(8, 24):
        left = 8 + (row % 2)
        right = 24 + (row % 2)
        jagged[row, left:right] = 1

    clean_sigma = mod._compute_single_mask_sigma_noise(clean)
    jagged_sigma = mod._compute_single_mask_sigma_noise(jagged)

    assert clean_sigma >= 0.0
    assert jagged_sigma > clean_sigma


def test_compute_single_mask_curvature_var_is_higher_for_jagged_mask() -> None:
    clean = np.zeros((32, 32), dtype=np.uint8)
    clean[8:24, 8:24] = 1

    jagged = np.zeros((32, 32), dtype=np.uint8)
    for row in range(8, 24):
        left = 8 + (row % 2)
        right = 24 + (row % 2)
        jagged[row, left:right] = 1

    clean_curvature = mod._compute_single_mask_curvature_var(clean)
    jagged_curvature = mod._compute_single_mask_curvature_var(jagged)

    assert clean_curvature >= 0.0
    assert jagged_curvature > clean_curvature


def test_compute_single_mask_ipr_and_solidity_capture_shape_roughness() -> None:
    clean = np.zeros((32, 32), dtype=np.uint8)
    clean[8:24, 8:24] = 1

    jagged = np.zeros((32, 32), dtype=np.uint8)
    for row in range(8, 24):
        left = 8 + (row % 2)
        right = 24 + (row % 2)
        jagged[row, left:right] = 1

    clean_ipr = mod._compute_single_mask_ipr(clean)
    jagged_ipr = mod._compute_single_mask_ipr(jagged)
    clean_solidity = mod._compute_single_mask_solidity(clean)
    jagged_solidity = mod._compute_single_mask_solidity(jagged)

    assert clean_ipr > 1.0
    assert jagged_ipr > clean_ipr
    np.testing.assert_allclose(clean_solidity, np.float32(1.0))
    assert jagged_solidity < clean_solidity


def test_prepare_refined_subject_run_creates_body_swim_editor_run(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)

    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    assert source.run_name == "subject_masks_001"
    assert refined.run_name == "refined_subject_masks_001"
    assert refined.component_names == ("subject_body", "swim_bladder")
    assert root["refined_subject_masks_runs"].attrs["latest"] == "refined_subject_masks_001"

    run = refined.group
    assert run.attrs["source_subject_mask_run"] == "subject_masks_001"
    assert run.attrs["source_crop_run"] == "crop_001"
    assert run.attrs["source_crop_storage_mode"] == "geometry_only"
    assert run.attrs["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert run.attrs["source_crop_revision"] == 4
    assert run.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"
    assert run.attrs["source_keypoint_group"] == "refined_keypoints_runs"
    assert run.attrs["source_keypoints_run"] == "refined_kp_001"
    assert run.attrs["source_keypoint_run"] == "refined_kp_001"
    assert run.attrs["label_schema_id"] == "refined_subject_v1_body_swim"
    assert run.attrs["git_commit"] == "b" * 40

    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    np.testing.assert_array_equal(masks[:, 0], np.asarray(source.masks_roi[:, 0], dtype=np.uint8))
    assert np.count_nonzero(masks[:, 1]) == 0
    assert tuple(int(v) for v in run["masks_roi"].chunks) == refined_subject_mask_storage_chunks(2, 8, 8)

    available = np.asarray(run["available_channels"][:], dtype=bool)
    np.testing.assert_array_equal(available, np.asarray([True, True], dtype=bool))
    edit_applied = np.asarray(run["edit_applied"][:], dtype=bool)
    assert not edit_applied.any()
    assert tuple(int(v) for v in run["edit_applied"].chunks) == (refined_subject_mask_metric_row_chunk(2), 1)

    metrics = run["metrics"]
    assert tuple(int(v) for v in metrics["mask_present"].chunks) == (refined_subject_mask_metric_row_chunk(2), 1)
    assert tuple(int(v) for v in metrics["bbox_xyxy"].chunks) == (refined_subject_mask_metric_row_chunk(2), 1, 4)
    mask_present = np.asarray(metrics["mask_present"][:], dtype=bool)
    area_px = np.asarray(metrics["area_px"][:], dtype=np.float32)
    centroid_xy = np.asarray(metrics["centroid_xy"][:], dtype=np.float32)
    centroid_valid = np.asarray(metrics["centroid_valid"][:], dtype=bool)
    bbox_xyxy = np.asarray(metrics["bbox_xyxy"][:], dtype=np.float32)
    bbox_valid = np.asarray(metrics["bbox_valid"][:], dtype=bool)
    np.testing.assert_array_equal(mask_present[:, 0], np.asarray([True, True], dtype=bool))
    np.testing.assert_array_equal(mask_present[:, 1], np.asarray([False, False], dtype=bool))
    np.testing.assert_allclose(area_px[:, 0], np.asarray([36.0, 16.0], dtype=np.float32))
    np.testing.assert_allclose(area_px[:, 1], np.asarray([0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(
        centroid_xy[:, 0, :],
        np.asarray([[3.5, 3.5], [3.5, 3.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(centroid_valid[:, 0], np.asarray([True, True], dtype=bool))
    np.testing.assert_array_equal(centroid_valid[:, 1], np.asarray([False, False], dtype=bool))
    np.testing.assert_allclose(
        bbox_xyxy[:, 0, :],
        np.asarray([[1.0, 1.0, 6.0, 6.0], [2.0, 2.0, 5.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(bbox_valid[:, 0], np.asarray([True, True], dtype=bool))
    np.testing.assert_array_equal(bbox_valid[:, 1], np.asarray([False, False], dtype=bool))

    body_reasons = read_reason_labels(run["components/subject_body"])
    swim_reasons = read_reason_labels(run["components/swim_bladder"])
    assert body_reasons is not None
    assert swim_reasons is not None
    assert body_reasons.tolist() == ["copied_from_source", "copied_from_source"]
    assert swim_reasons.tolist() == ["clean", "clean"]

    body_group = run["components/subject_body"]
    body_provenance = body_group["provenance"]
    np.testing.assert_array_equal(
        np.asarray(body_group["manual_override"][:], dtype=bool),
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(body_group["source_row_stale"][:], dtype=bool),
        np.asarray([False, False], dtype=bool),
    )
    assert body_group.attrs["component_schema_id"] == "subject_body_v1"
    assert body_group.attrs["anatomical_scope"] == "body_core"
    assert body_group.attrs["pectoral_fin_policy"] == "excluded_or_unresolved"
    assert body_provenance.attrs["source_stage"] == "subject_mask_runs"
    assert body_provenance.attrs["source_run"] == "subject_masks_001"
    assert body_provenance.attrs["source_method"] == "subject_mask_threshold_lr_v1"
    assert body_provenance.attrs["source_crop_run"] == "crop_001"
    assert body_provenance.attrs["source_crop_storage_mode"] == "geometry_only"
    assert body_provenance.attrs["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert body_provenance.attrs["source_crop_revision"] == 4
    assert body_provenance.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"
    assert body_provenance.attrs["source_channels"] == ["subject_body"]
    assert body_provenance.attrs["source_label_schema_id"] == "subject_v1_lr"
    assert body_provenance.attrs["last_update_stage"] == mod.REFINED_SUBJECT_STAGE_NAME
    assert body_provenance.attrs["last_update_mode"] == "create"
    assert body_provenance.attrs["last_update_method"] == mod.DEFAULT_RUN_METHOD
    assert body_provenance.attrs["updated_at_utc"] == run.attrs["created_at_utc"]

    swim_provenance = run["components/swim_bladder/provenance"]
    assert swim_provenance.attrs["source_stage"] == "subject_mask_runs"
    assert swim_provenance.attrs["source_run"] == "subject_masks_001"
    assert swim_provenance.attrs["source_method"] == "subject_mask_threshold_lr_v1"
    assert swim_provenance.attrs["source_crop_run"] == "crop_001"
    assert swim_provenance.attrs["source_crop_storage_mode"] == "geometry_only"
    assert swim_provenance.attrs["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert swim_provenance.attrs["source_crop_revision"] == 4
    assert swim_provenance.attrs["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"
    assert swim_provenance.attrs["source_channels"] == ["swim_bladder"]
    assert swim_provenance.attrs["source_label_schema_id"] == "subject_v1_lr"
    assert swim_provenance.attrs["last_update_mode"] == "create"

    body_metrics = run["components/subject_body/metrics"]
    swim_metrics = run["components/swim_bladder/metrics"]
    np.testing.assert_array_equal(
        np.asarray(body_metrics["component_count"][:], dtype=np.int32),
        np.asarray([1, 1], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(body_metrics["largest_component_fraction"][:], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(body_metrics["hole_count"][:], dtype=np.int32),
        np.asarray([0, 0], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(body_metrics["hole_area_fraction"][:], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    assert np.all(np.asarray(body_metrics["sigma_noise"][:], dtype=np.float32) > 0.0)
    assert np.all(np.asarray(body_metrics["curvature_var"][:], dtype=np.float32) > 0.0)
    assert np.all(np.asarray(body_metrics["ipr"][:], dtype=np.float32) > 1.0)
    np.testing.assert_allclose(
        np.asarray(body_metrics["solidity"][:], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(swim_metrics["component_count"][:], dtype=np.int32),
        np.asarray([0, 0], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(swim_metrics["sigma_noise"][:], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(swim_metrics["curvature_var"][:], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(swim_metrics["ipr"][:], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(swim_metrics["solidity"][:], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )

    provenance = run.attrs["provenance"]
    assert provenance["stage"] == "refine_subject_masks"
    assert provenance["command"] == "scripts/py -m fisheye.tune.refined_subject_mask_review"
    assert provenance["git"]["commit"] == "b" * 40
    assert provenance["inputs"]["source_subject_mask_run"] == "subject_masks_001"
    assert provenance["inputs"]["source_crop_storage_mode"] == "geometry_only"
    assert provenance["inputs"]["source_crop_signature"] == "{'signature_version': 2, 'crop_revision': 4}"
    assert provenance["inputs"]["source_crop_revision"] == 4
    assert provenance["inputs"]["source_detect_review_status_ref"] == "refined_detect_runs/refined_detect_001/review_status"
    assert provenance["inputs"]["source_keypoints_run"] == "refined_kp_001"
    assert provenance["inputs"]["source_keypoint_group"] == "refined_keypoints_runs"


def test_prepare_existing_refined_subject_run_skips_eager_component_metric_backfill(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)

    mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    def fail_component_metric_backfill(_masks: np.ndarray) -> dict[str, np.ndarray]:
        raise AssertionError("existing refined runs should not eagerly recompute component metrics")

    monkeypatch.setattr(mod, "_compute_component_topology_metrics", fail_component_metric_backfill)

    _source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body",),
    )

    assert refined.run_name == "refined_subject_masks_001"
    assert refined.component_names == ("subject_body", "swim_bladder")


def test_prepare_refined_subject_run_defaults_to_available_source_components(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)

    _source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
    )

    assert refined.component_names == ("subject_body", "eye_left", "eye_right")


def test_load_refined_component_source_runs_falls_back_to_coarse_subject_source_for_legacy_lineage(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)
    subject = root["subject_mask_runs"]["subject_masks_001"]
    subject.attrs["component_provenance"] = {
        "components": {
            "eye_left": {
                "source_stage": "refined_eye_masks_runs",
                "source_run": "refined_eye_masks_001",
                "source_channels": ["eye_left"],
            },
            "eye_right": {
                "source_stage": "refined_eye_masks_runs",
                "source_run": "refined_eye_masks_001",
                "source_channels": ["eye_right"],
            },
        }
    }

    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("eye_left", "eye_right"),
    )

    primary_source, component_sources = mod._load_refined_component_source_runs(  # noqa: SLF001
        root,
        refined,
        default_source=source,
    )

    assert primary_source.run_name == "subject_masks_001"
    assert component_sources["eye_left"].run_name == "subject_masks_001"
    assert component_sources["eye_right"].run_name == "subject_masks_001"
    assert refined.group["components/eye_left/provenance"].attrs["source_stage"] == "refined_eye_masks_runs"
    assert refined.group["components/eye_right/provenance"].attrs["source_stage"] == "refined_eye_masks_runs"


def test_save_refined_subject_roi_updates_edit_applied_metrics_and_reasons() -> None:
    root = _build_subject_review_root()
    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    edited[1, 4:6, 4:6] = 1

    mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
    )

    run = refined.group
    saved = np.asarray(run["masks_roi"][0], dtype=np.uint8)
    np.testing.assert_array_equal(saved, edited)

    edit_applied = np.asarray(run["edit_applied"][0], dtype=bool)
    np.testing.assert_array_equal(edit_applied, np.asarray([True, True], dtype=bool))

    mask_present = np.asarray(run["metrics/mask_present"][0], dtype=bool)
    area_px = np.asarray(run["metrics/area_px"][0], dtype=np.float32)
    centroid_xy = np.asarray(run["metrics/centroid_xy"][0], dtype=np.float32)
    centroid_valid = np.asarray(run["metrics/centroid_valid"][0], dtype=bool)
    bbox_xyxy = np.asarray(run["metrics/bbox_xyxy"][0], dtype=np.float32)
    bbox_valid = np.asarray(run["metrics/bbox_valid"][0], dtype=bool)
    np.testing.assert_array_equal(mask_present, np.asarray([False, True], dtype=bool))
    np.testing.assert_allclose(area_px, np.asarray([0.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(
        centroid_xy,
        np.asarray([[0.0, 0.0], [4.5, 4.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(centroid_valid, np.asarray([False, True], dtype=bool))
    np.testing.assert_allclose(
        bbox_xyxy,
        np.asarray([[0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 5.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(bbox_valid, np.asarray([False, True], dtype=bool))

    body_group = run["components/subject_body"]
    swim_group = run["components/swim_bladder"]
    body_provenance = body_group["provenance"]
    swim_provenance = swim_group["provenance"]
    assert bool(np.asarray(body_group["edit_applied"][0], dtype=bool)) is True
    assert bool(np.asarray(swim_group["edit_applied"][0], dtype=bool)) is True
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is True
    assert bool(np.asarray(swim_group["manual_override"][0], dtype=bool)) is True
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is False
    body_reasons = read_reason_labels(body_group)
    swim_reasons = read_reason_labels(swim_group)
    assert body_reasons is not None
    assert swim_reasons is not None
    assert body_reasons[0] == "manual_correction"
    assert swim_reasons[0] == "manual_correction"
    body_contours = body_group["contours"]
    swim_contours = swim_group["contours"]
    assert body_contours.attrs["cache_coverage"] == "partial_row_updates"
    assert swim_contours.attrs["cache_coverage"] == "partial_row_updates"
    assert int(body_contours["ptr"][0]) == -1
    assert int(body_contours["len"][0]) == 0
    assert int(swim_contours["ptr"][0]) == 0
    assert int(swim_contours["len"][0]) > 0
    assert int(body_group["row_revision"][0]) == 1
    assert int(swim_group["row_revision"][0]) == 1
    assert int(body_group["row_revision"][1]) == 0
    assert int(swim_group["row_revision"][1]) == 0
    np.testing.assert_allclose(
        np.asarray(body_group["metrics/largest_component_fraction"][0], dtype=np.float32),
        np.float32(0.0),
    )
    np.testing.assert_allclose(
        np.asarray(body_group["metrics/sigma_noise"][0], dtype=np.float32),
        np.float32(0.0),
    )
    np.testing.assert_allclose(
        np.asarray(body_group["metrics/curvature_var"][0], dtype=np.float32),
        np.float32(0.0),
    )
    np.testing.assert_allclose(
        np.asarray(body_group["metrics/ipr"][0], dtype=np.float32),
        np.float32(0.0),
    )
    np.testing.assert_allclose(
        np.asarray(body_group["metrics/solidity"][0], dtype=np.float32),
        np.float32(0.0),
    )
    np.testing.assert_array_equal(
        np.asarray(body_group["metrics/component_count"][0], dtype=np.int32),
        np.int32(0),
    )
    np.testing.assert_array_equal(
        np.asarray(swim_group["metrics/component_count"][0], dtype=np.int32),
        np.int32(1),
    )
    np.testing.assert_allclose(
        np.asarray(swim_group["metrics/largest_component_fraction"][0], dtype=np.float32),
        np.float32(1.0),
    )
    assert float(np.asarray(swim_group["metrics/sigma_noise"][0], dtype=np.float32)) > 0.0
    assert float(np.asarray(swim_group["metrics/curvature_var"][0], dtype=np.float32)) == 0.0
    assert float(np.asarray(swim_group["metrics/ipr"][0], dtype=np.float32)) > 1.0
    np.testing.assert_allclose(
        np.asarray(swim_group["metrics/solidity"][0], dtype=np.float32),
        np.float32(1.0),
    )
    assert body_provenance.attrs["last_update_stage"] == mod.REFINED_SUBJECT_STAGE_NAME
    assert body_provenance.attrs["last_update_mode"] == "interactive"
    assert body_provenance.attrs["last_update_method"] == mod.DEFAULT_RUN_METHOD
    assert body_provenance.attrs["updated_at_utc"] == run.attrs["updated_at_utc"]
    assert swim_provenance.attrs["last_update_mode"] == "interactive"
    assert swim_provenance.attrs["last_update_method"] == mod.DEFAULT_RUN_METHOD
    assert swim_provenance.attrs["updated_at_utc"] == run.attrs["updated_at_utc"]


def test_save_refined_subject_roi_can_scope_updates_to_requested_component() -> None:
    root = _build_subject_review_root()
    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    edited[1, 4:6, 4:6] = 1

    mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
        component_names=("subject_body",),
    )

    run = refined.group
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][0, 0], dtype=np.uint8),
        np.asarray(edited[0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][0, 1], dtype=np.uint8),
        np.zeros((8, 8), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][0], dtype=bool),
        np.asarray([True, False], dtype=bool),
    )

    body_group = run["components/subject_body"]
    swim_group = run["components/swim_bladder"]
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is True
    assert bool(np.asarray(swim_group["manual_override"][0], dtype=bool)) is False
    assert body_group["provenance"].attrs["last_update_mode"] == "interactive"
    assert swim_group["provenance"].attrs["last_update_mode"] == "create"


def test_save_refined_subject_roi_scoped_save_passes_only_active_component_to_finalize(monkeypatch) -> None:
    root = _build_subject_review_root()
    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "eye_left", "eye_right"),
    )

    calls: list[tuple[str, ...] | None] = []

    def _capture_materialize(_refined, *, updated_components=None):  # noqa: ANN001
        if updated_components is None:
            calls.append(None)
        else:
            calls.append(tuple(str(name) for name in updated_components))

    monkeypatch.setattr(mod, "_materialize_refined_eye_compat_if_needed", _capture_materialize)

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0

    mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
        component_names=("subject_body",),
    )

    assert calls == [("subject_body",)]


def test_check_refined_subject_source_updates_auto_syncs_unedited_rows(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    _source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    subject = root["subject_mask_runs"]["subject_masks_001"]
    source_masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    source_masks[0, 0, 1:7, 1:7] = 0
    source_masks[0, 0, 2:6, 2:6] = 1
    subject["masks_roi"][:] = source_masks

    summary = mod.check_refined_subject_source_updates(
        "/tmp/fake_subject_review.zarr",
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
    )

    assert summary["status"] == "updated"
    assert summary["source_changed_roi_count"] == 1
    assert summary["auto_synced_roi_count"] == 1
    assert summary["stale_marked_roi_count"] == 0
    assert summary["auto_synced_roi_indices"] == [0]

    run = refined.group
    body_group = run["components/subject_body"]
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][0, 0], dtype=np.uint8),
        np.asarray(subject["masks_roi"][0, 0], dtype=np.uint8),
    )
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is False
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is False


def test_check_refined_subject_source_updates_marks_manual_rows_stale(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
    )

    subject = root["subject_mask_runs"]["subject_masks_001"]
    source_masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    source_masks[0, 0, 2:6, 2:6] = 0
    source_masks[0, 0, 3:5, 3:5] = 1
    subject["masks_roi"][:] = source_masks

    summary = mod.check_refined_subject_source_updates(
        "/tmp/fake_subject_review.zarr",
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
    )

    assert summary["status"] == "updated"
    assert summary["source_changed_roi_count"] == 1
    assert summary["auto_synced_roi_count"] == 0
    assert summary["stale_marked_roi_count"] == 1
    assert summary["stale_roi_indices"] == [0]
    assert summary["stale_total"] == 1

    run = mod._open_existing_refined_subject_run(root, "refined_subject_masks_001").group  # noqa: SLF001
    body_group = run["components/subject_body"]
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is True
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is True
    assert body_group.attrs["source_update_pending_rows"] == [0]
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][0, 0], dtype=np.uint8),
        np.asarray(edited[0], dtype=np.uint8),
    )
    review_payload = dict(run.attrs.get("component_review_statuses") or {}).get("subject_body", {})
    assert review_payload["state"] == "needs_review"


def test_check_refined_subject_source_updates_legacy_bootstrap_auto_syncs_unedited_rows(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    _source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    body_group = refined.group["components/subject_body"]
    del body_group["source_row_fingerprint"]
    del body_group["manual_override"]
    del body_group["source_row_stale"]
    body_group.attrs.pop("source_sync_schema_id", None)

    subject = root["subject_mask_runs"]["subject_masks_001"]
    source_masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    source_masks[0, 0, 2:6, 2:6] = 0
    source_masks[0, 0, 3:5, 3:5] = 1
    subject["masks_roi"][:] = source_masks

    summary = mod.check_refined_subject_source_updates(
        "/tmp/fake_subject_review.zarr",
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
        assume_source_changed_untracked=True,
    )

    assert summary["status"] == "updated"
    assert summary["assume_source_changed_untracked"] is True
    assert summary["source_changed_roi_count"] == 1
    assert summary["auto_synced_roi_count"] == 1
    assert summary["stale_marked_roi_count"] == 0
    assert summary["unchanged_roi_count"] == 0

    run = mod._open_existing_refined_subject_run(root, "refined_subject_masks_001").group  # noqa: SLF001
    body_group = run["components/subject_body"]
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][0, 0], dtype=np.uint8),
        np.asarray(subject["masks_roi"][0, 0], dtype=np.uint8),
    )
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is False
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is False


def test_check_refined_subject_source_updates_legacy_bootstrap_marks_manual_rows_stale(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
    )

    body_group = refined.group["components/subject_body"]
    del body_group["source_row_fingerprint"]
    del body_group["manual_override"]
    del body_group["source_row_stale"]
    body_group.attrs.pop("source_sync_schema_id", None)

    subject = root["subject_mask_runs"]["subject_masks_001"]
    source_masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    source_masks[0, 0, 2:6, 2:6] = 0
    source_masks[0, 0, 3:5, 3:5] = 1
    subject["masks_roi"][:] = source_masks

    summary = mod.check_refined_subject_source_updates(
        "/tmp/fake_subject_review.zarr",
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
        assume_source_changed_untracked=True,
    )

    assert summary["status"] == "updated"
    assert summary["assume_source_changed_untracked"] is True
    assert summary["source_changed_roi_count"] == 1
    assert summary["auto_synced_roi_count"] == 0
    assert summary["stale_marked_roi_count"] == 1
    assert summary["stale_roi_indices"] == [0]

    run = mod._open_existing_refined_subject_run(root, "refined_subject_masks_001").group  # noqa: SLF001
    body_group = run["components/subject_body"]
    np.testing.assert_array_equal(
        np.asarray(run["masks_roi"][0, 0], dtype=np.uint8),
        np.asarray(edited[0], dtype=np.uint8),
    )
    assert bool(np.asarray(body_group["manual_override"][0], dtype=bool)) is True
    assert bool(np.asarray(body_group["source_row_stale"][0], dtype=bool)) is True


def test_check_refined_subject_source_updates_force_source_changed_reprocesses_seeded_rows(monkeypatch) -> None:
    root = _build_subject_review_root()
    _patch_review_provenance(monkeypatch)
    monkeypatch.setattr(mod, "open_zarr_root", lambda *_args, **_kwargs: root)

    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited = np.asarray(refined.group["masks_roi"][0], dtype=np.uint8)
    edited[0, 1:7, 1:7] = 0
    mod.save_refined_subject_roi(
        source=source,
        refined=refined,
        roi_idx=0,
        edited_masks=edited,
    )

    subject = root["subject_mask_runs"]["subject_masks_001"]
    source_masks = np.asarray(subject["masks_roi"][:], dtype=np.uint8)
    source_masks[0, 0, 2:6, 2:6] = 0
    source_masks[0, 0, 3:5, 3:5] = 1
    subject["masks_roi"][:] = source_masks

    baseline_summary = mod.check_refined_subject_source_updates(
        "/tmp/fake_subject_review.zarr",
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
        assume_source_changed_untracked=True,
    )
    assert baseline_summary["source_changed_roi_count"] == 1
    assert baseline_summary["stale_marked_roi_count"] == 1

    rerun_summary = mod.check_refined_subject_source_updates(
        "/tmp/fake_subject_review.zarr",
        refined_run="refined_subject_masks_001",
        component_name="subject_body",
        roi_indices=[0],
        force_source_changed=True,
    )

    assert rerun_summary["status"] == "updated"
    assert rerun_summary["force_source_changed"] is True
    assert rerun_summary["source_changed_roi_count"] == 1
    assert rerun_summary["auto_synced_roi_count"] == 0
    assert rerun_summary["stale_marked_roi_count"] == 1


def test_apply_refined_subject_roi_rows_updates_only_requested_component() -> None:
    root = _build_subject_review_root()
    source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    edited_batch = np.stack(
        [
            np.asarray(refined.group["masks_roi"][0], dtype=np.uint8),
            np.asarray(refined.group["masks_roi"][1], dtype=np.uint8),
        ],
        axis=0,
    )
    edited_batch[0, 0, 1:7, 1:7] = 0
    edited_batch[0, 1, 4:6, 4:6] = 1
    edited_batch[1, 1, 3:5, 3:5] = 1

    normalized_rows = mod._apply_refined_subject_roi_rows(
        source=source,
        refined=refined,
        roi_indices=[0, 1],
        edited_masks_batch=edited_batch,
        component_names=("subject_body",),
    )

    assert normalized_rows == (0, 1)

    run = refined.group
    saved = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    np.testing.assert_array_equal(saved[0, 0], np.zeros((8, 8), dtype=np.uint8))
    np.testing.assert_array_equal(saved[1, 0], np.asarray(source.masks_roi[1, 0], dtype=np.uint8))
    np.testing.assert_array_equal(saved[:, 1], np.zeros((2, 8, 8), dtype=np.uint8))

    np.testing.assert_array_equal(
        np.asarray(run["edit_applied"][:], dtype=bool),
        np.asarray(
            [
                [True, False],
                [False, False],
            ],
            dtype=bool,
        ),
    )

    body_group = run["components/subject_body"]
    swim_group = run["components/swim_bladder"]
    body_provenance = body_group["provenance"]
    swim_provenance = swim_group["provenance"]
    body_reasons = read_reason_labels(body_group)
    swim_reasons = read_reason_labels(swim_group)
    assert body_reasons is not None
    assert swim_reasons is not None
    assert body_reasons.tolist() == ["manual_correction", "copied_from_source"]
    assert swim_reasons.tolist() == ["clean", "clean"]
    np.testing.assert_allclose(
        np.asarray(swim_group["area_px"][:], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
    )
    assert body_provenance.attrs["last_update_mode"] == "interactive"
    assert body_provenance.attrs["last_update_method"] == mod.DEFAULT_RUN_METHOD
    assert body_provenance.attrs["updated_at_utc"] == run.attrs["updated_at_utc"]
    assert swim_provenance.attrs["last_update_mode"] == "create"


def test_format_component_summary_lines_exposes_common_geometry_and_qc() -> None:
    root = _build_subject_review_root()
    _source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    body_group = refined.group["components/subject_body"]
    lines = mod._format_component_summary_lines(
        refined.group,
        body_group,
        comp_idx=0,
        roi_idx=0,
    )

    assert any("area_px=36.0" in line for line in lines)
    assert any("centroid=(3.5, 3.5)" in line for line in lines)
    assert any("bbox=[1.0,1.0,6.0,6.0]" in line for line in lines)
    assert any("components=1" in line for line in lines)
    assert any("sigma_noise=" in line for line in lines)
    assert any("curvature_var=" in line for line in lines)
    assert any("ipr=" in line for line in lines)
    assert any("solidity=" in line for line in lines)
    assert any("reason=copied_from_source" in line for line in lines)


def test_apply_component_review_status_aggregates_run_review_state() -> None:
    root = _build_subject_review_root()
    _source, refined = mod.prepare_refined_subject_run(
        root,
        subject_run="subject_masks_001",
        refined_run="refined_subject_masks_001",
        components=("subject_body", "swim_bladder"),
    )

    body_payload, run_payload = mod.apply_component_review_status(
        refined.parent,
        refined.run_name,
        refined.group,
        component_name="subject_body",
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="body looks good",
    )
    assert body_payload["state"] == "approved"
    assert run_payload["state"] == "pending"

    swim_payload, run_payload = mod.apply_component_review_status(
        refined.parent,
        refined.run_name,
        refined.group,
        component_name="swim_bladder",
        state="needs_review",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="swim bladder not done",
    )
    assert swim_payload["state"] == "needs_review"
    assert run_payload["state"] == "needs_review"

    _swim_payload, run_payload = mod.apply_component_review_status(
        refined.parent,
        refined.run_name,
        refined.group,
        component_name="swim_bladder",
        state="approved",
        method="manual",
        intended_use="training",
        reviewer="tester",
        notes="done",
    )
    assert run_payload["state"] == "approved"
    assert refined.parent.attrs["refined_subject_mask_review_status_latest"] == refined.run_name
