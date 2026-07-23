from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fisheye.shared.mask_store import open_mask_store, write_component_rle_mask_store_from_dense
import fisheye.visualization.visualize_subject_shape_overlays as overlay_mod
from fisheye.visualization.visualize_subject_shape_overlays import (
    SubjectShapeOverlayContext,
    export_subject_shape_overlays,
    open_subject_shape_overlay_context,
    render_subject_shape_overlay,
)


def _disk_mask(height: int, width: int, center_y: float, center_x: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2).astype(np.uint8)


def _build_refined_root(store_path: Path) -> zarr.Group:
    root = zarr.open_group(str(store_path), mode="w")
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    run.attrs.update(
        {
            "mask_labels": labels,
            "label_schema_id": "subject_v1_lr",
            "component_metrics_schema_id": "refined_subject_component_mask_metrics_v1",
            "component_review_statuses": {
                component: {"state": "approved", "method": "unit_test"}
                for component in labels
            },
        }
    )
    run.create_array("available_channels", data=np.asarray([True, True, True, True], dtype=bool), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    run.create_array("source_refined_row_ids", data=np.asarray([100, 101], dtype=np.int64), overwrite=True)

    masks = np.zeros((2, 4, 20, 20), dtype=np.uint8)
    for row_idx in range(2):
        masks[row_idx, 0, 3:17, 7:13] = 1
        masks[row_idx, 1] = _disk_mask(20, 20, 6 + row_idx, 7, 2.0)
        masks[row_idx, 2] = _disk_mask(20, 20, 6 + row_idx, 12, 2.0)
        masks[row_idx, 3] = _disk_mask(20, 20, 12, 10, 2.0)
    run.create_array("masks_roi", data=masks, chunks=(1, 1, 20, 20), overwrite=True)
    return root


def _replace_refined_masks_with_rle(root: zarr.Group) -> None:
    run = root["refined_subject_masks_runs"]["refined_001"]
    dense = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    del run["masks_roi"]
    write_component_rle_mask_store_from_dense(
        run,
        dense,
        component_names=[str(label) for label in run.attrs["mask_labels"]],
        encode_row_chunk_size=1,
    )


def _add_persisted_eye_left_contours(root: zarr.Group) -> None:
    run = root["refined_subject_masks_runs"]["refined_001"]
    contours = run.require_group("components").require_group("eye_left").require_group("contours")
    contours.attrs["contour_schema_id"] = "component_contours_v1"
    contours.attrs["coordinate_space"] = "roi_pixels"
    contours.create_array("ptr", data=np.asarray([0, -1], dtype=np.int64), overwrite=True)
    contours.create_array("len", data=np.asarray([5, 0], dtype=np.int32), overwrite=True)
    contours.create_array(
        "points_xy",
        data=np.asarray(
            [
                [5.0, 5.0],
                [9.0, 5.0],
                [9.0, 9.0],
                [5.0, 9.0],
                [5.0, 5.0],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )


def _build_overlay_context(
    store_path: Path,
    *,
    compact_masks: bool = False,
    persisted_contours: bool = False,
    branched_body: bool = False,
) -> SubjectShapeOverlayContext:
    """Build a renderer-only context with explicit camera-to-ROI placement."""

    root = _build_refined_root(store_path)
    refined = root["refined_subject_masks_runs/refined_001"]
    if branched_body:
        masks = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
        masks[0, 0] = 0
        masks[0, 0, 3:17, 10] = 1
        masks[0, 0, 10, 10:18] = 1
        refined["masks_roi"][:] = masks
    if persisted_contours:
        _add_persisted_eye_left_contours(root)
    if compact_masks:
        _replace_refined_masks_with_rle(root)

    analysis = root.require_group("analysis")
    parent = analysis.require_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_001"
    shape = parent.require_group("shape_001")
    shape.attrs["source_refined_subject_masks_run"] = "refined_001"
    offsets = np.asarray([[30.0, 40.0], [35.0, 45.0]], dtype=np.float32)

    row_index = shape.require_group("row_index")
    row_index.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32))
    components = shape.require_group("components")
    body = components.require_group("subject_body")
    swim = components.require_group("swim_bladder")
    roi_centerline = np.asarray(
        [[10.0, 4.0], [10.0, 8.0], [10.0, 12.0], [10.0, 16.0]],
        dtype=np.float32,
    )
    centerline = np.stack([roi_centerline + offset for offset in offsets], axis=0)
    control_roi = np.asarray(
        [[10.0, 4.0], [11.0, 10.0], [10.0, 16.0]],
        dtype=np.float32,
    )
    control = np.stack([control_roi + offset for offset in offsets], axis=0)
    tail_roi = np.asarray(
        [[10.0, 10.0], [10.0, 13.0], [10.0, 16.0]],
        dtype=np.float32,
    )
    tail = np.stack([tail_roi + offset for offset in offsets], axis=0)

    body.create_array("centerline_xy", data=centerline)
    body.create_array("centerline_valid", data=np.asarray([True, True], dtype=bool))
    body.create_array("bspline_sample_xy", data=centerline)
    body.create_array("bspline_control_points_xy", data=control)
    body.create_array("tail_sample_xy", data=tail)
    body.create_array(
        "tail_normal_xy",
        data=np.repeat(
            np.asarray([[[1.0, 0.0]]], dtype=np.float32),
            2 * tail.shape[1],
            axis=0,
        ).reshape(2, tail.shape[1], 2),
    )
    for name in (
        "snout_tip_xy",
        "head_endpoint_xy",
        "tail_base_xy",
        "tail_tip_xy",
    ):
        roi_point = {
            "snout_tip_xy": [10.0, 4.0],
            "head_endpoint_xy": [10.0, 5.0],
            "tail_base_xy": [10.0, 10.0],
            "tail_tip_xy": [10.0, 16.0],
        }[name]
        body.create_array(
            name,
            data=offsets + np.asarray(roi_point, dtype=np.float32),
        )
    for name in (
        "snout_tip_valid",
        "tail_base_valid",
        "bspline_valid",
        "tail_sample_valid",
        "centerline_reaches_snout",
    ):
        body.create_array(name, data=np.asarray([True, True], dtype=bool))
    for name, values in (
        ("body_arclength_px", [12.0, 12.0]),
        ("tail_segment_arclength_px", [6.0, 6.0]),
        ("bspline_arc_length_px", [12.0, 12.0]),
        ("head_endpoint_to_snout_distance_px", [1.0, 1.0]),
    ):
        body.create_array(name, data=np.asarray(values, dtype=np.float32))
    zero_reasons = np.zeros((2, 64), dtype=np.uint8)
    for name in (
        "centerline_failure_reason_bytes",
        "snout_tip_failure_reason_bytes",
        "centerline_snout_check_reason_bytes",
        "tail_base_failure_reason_bytes",
        "bspline_failure_reason_bytes",
        "tail_sample_failure_reason_bytes",
    ):
        body.create_array(name, data=zero_reasons)

    swim.create_array(
        "caudal_contour_point_xy",
        data=offsets + np.asarray([10.0, 14.0], dtype=np.float32),
    )
    swim.create_array("caudal_contour_valid", data=np.asarray([True, True], dtype=bool))
    swim.create_array("caudal_contour_failure_reason_bytes", data=zero_reasons)

    frame = shape.require_group("body_frame")
    frame.create_array(
        "origin_xy",
        data=offsets + np.asarray([10.0, 10.0], dtype=np.float32),
    )
    frame.create_array("axis_valid", data=np.asarray([True, True], dtype=bool))
    frame.create_array(
        "forward_axis_xy",
        data=np.asarray([[0.0, -1.0], [0.0, -1.0]], dtype=np.float32),
    )
    frame.create_array(
        "left_axis_xy",
        data=np.asarray([[-1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
    )

    return SubjectShapeOverlayContext(
        root=root,
        shape_run_name="shape_001",
        shape_group=shape,
        refined_run_name="refined_001",
        refined_group=refined,
        label_map={
            str(label): idx
            for idx, label in enumerate(refined.attrs["mask_labels"])
        },
        mask_store=open_mask_store(refined, prefer="dense"),
        coordinate_publication=object(),  # type: ignore[arg-type]
        source_camera_offset_xy=offsets.astype(np.float64),
        source_crop_row_ids=np.asarray([0, 1], dtype=np.int64),
        crop_source=None,
    )


def test_overlay_crop_run_must_equal_canonical_source() -> None:
    publication = SimpleNamespace(
        source=SimpleNamespace(
            context=SimpleNamespace(
                source=SimpleNamespace(
                    source=SimpleNamespace(crop_path="crop_runs/crop_exact")
                )
            )
        )
    )

    assert overlay_mod._canonical_crop_run_name(publication, None) == "crop_exact"
    assert (
        overlay_mod._canonical_crop_run_name(
            publication,
            "crop_runs/crop_exact",
        )
        == "crop_exact"
    )
    with pytest.raises(ValueError, match="exact crop authority"):
        overlay_mod._canonical_crop_run_name(publication, "crop_other")


def test_base_image_uses_reordered_source_crop_row_ids(tmp_path: Path) -> None:
    class FakeCropSource:
        total_rois = 6
        crop_run_name = "crop_exact"

        def __init__(self) -> None:
            self.rows: list[int] = []

        def __getitem__(self, row: int) -> np.ndarray:
            self.rows.append(int(row))
            return np.full((2, 2), float(row), dtype=np.float32)

    ctx = _build_overlay_context(tmp_path / "shape.zarr")
    crop_source = FakeCropSource()
    reordered = SubjectShapeOverlayContext(
        **{
            **ctx.__dict__,
            "source_crop_row_ids": np.asarray([5, 2], dtype=np.int64),
            "crop_source": crop_source,
        }
    )

    image, label = overlay_mod._base_image(reordered, 1, None)

    assert crop_source.rows == [2]
    assert label == "crop:crop_exact:row=2"
    assert image.shape == (2, 2)


def test_open_overlay_context_rejects_unpublished_subject_shape_run(tmp_path: Path) -> None:
    zarr_path = tmp_path / "legacy-shape.zarr"
    root = _build_refined_root(zarr_path)
    parent = root.require_group("analysis").require_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_legacy"
    parent.require_group("shape_legacy")

    with pytest.raises(ValueError, match="not a valid canonical coordinate publication"):
        open_subject_shape_overlay_context(
            zarr_path,
            shape_run="shape_legacy",
            use_crop_images=False,
        )


def test_render_subject_shape_overlay_from_mask_background(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "shape.zarr"
    ctx = _build_overlay_context(zarr_path)
    fig = render_subject_shape_overlay(ctx, row=0, show_skeleton=True)

    assert fig.axes
    assert "Subject shape overlay" in fig.axes[0].get_title()
    labels = [collection.get_label() for collection in fig.axes[0].collections]
    assert "body skeleton" in labels
    assert "snout tip" in labels
    centerline = next(line for line in fig.axes[0].lines if line.get_label() == "centerline")
    np.testing.assert_allclose(centerline.get_xdata(), [10.0, 10.0, 10.0, 10.0])
    np.testing.assert_allclose(centerline.get_ydata(), [4.0, 8.0, 12.0, 16.0])
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_reads_compact_mask_store(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "shape.zarr"
    ctx = _build_overlay_context(zarr_path, compact_masks=True)
    fig = render_subject_shape_overlay(ctx, row=0, show_skeleton=True)

    assert "masks_roi" not in ctx.refined_group
    assert ctx.mask_store.encoding == "component_rle_v1"
    labels = [collection.get_label() for collection in fig.axes[0].collections]
    assert "body skeleton" in labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_can_draw_persisted_eye_contours(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "shape.zarr"
    ctx = _build_overlay_context(zarr_path, persisted_contours=True)
    fig = render_subject_shape_overlay(ctx, row=0, contour_source="persisted")

    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "eye left persisted" in labels
    assert "body contour" not in labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_can_offset_skeleton(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "shape.zarr"
    ctx = _build_overlay_context(zarr_path)
    fig = render_subject_shape_overlay(ctx, row=0, show_skeleton=True, skeleton_style="offset")

    labels = [collection.get_label() for collection in fig.axes[0].collections]
    assert "body skeleton offset" in labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_can_draw_spline_layers(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "shape.zarr"
    ctx = _build_overlay_context(zarr_path)
    fig = render_subject_shape_overlay(
        ctx,
        row=0,
        show_bspline=True,
        show_spline_control_points=True,
        show_tail_samples=True,
        show_tail_normals=True,
    )

    line_labels = [line.get_label() for line in fig.axes[0].lines]
    collection_labels = [collection.get_label() for collection in fig.axes[0].collections]
    assert "B-spline sample" in line_labels
    assert "tail normals" in line_labels
    assert "B-spline control points" in collection_labels
    assert "tail samples" in collection_labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_can_highlight_unused_skeleton_branches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "shape.zarr"
    ctx = _build_overlay_context(zarr_path, branched_body=True)
    fig = render_subject_shape_overlay(ctx, row=0, show_skeleton=True, skeleton_style="branches")

    labels = [collection.get_label() for collection in fig.axes[0].collections]
    assert "body skeleton all" in labels
    assert "unused skeleton branches" in labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_compare_draws_mask_and_persisted_contours(
    tmp_path: Path,
    monkeypatch,
) -> None:
    zarr_path = tmp_path / "shape.zarr"
    ctx = _build_overlay_context(zarr_path, persisted_contours=True)
    fig = render_subject_shape_overlay(ctx, row=0, contour_source="compare")

    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "eye left from mask" in labels
    assert "eye left persisted" in labels
    assert "body contour from mask" in labels
    fig.canvas.draw()
    plt.close(fig)


def test_export_subject_shape_overlay_png(tmp_path: Path, monkeypatch) -> None:
    zarr_path = tmp_path / "shape.zarr"
    ctx = _build_overlay_context(zarr_path)
    monkeypatch.setattr(
        overlay_mod,
        "open_subject_shape_overlay_context",
        lambda *_args, **_kwargs: ctx,
    )

    paths = export_subject_shape_overlays(
        zarr_path,
        output_dir=tmp_path / "overlays",
        shape_run="shape_001",
        rows=[0, 1],
        use_crop_images=False,
        dpi=80,
        contour_source="auto",
        show_skeleton=True,
    )

    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
