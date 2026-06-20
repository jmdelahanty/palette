from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fisheye.analysis import subject_shape_runs
from fisheye.shared.mask_store import write_component_rle_mask_store_from_dense
from fisheye.visualization.visualize_subject_shape_overlays import (
    export_subject_shape_overlays,
    open_subject_shape_overlay_context,
    render_subject_shape_overlay,
)


def _patch_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        subject_shape_runs,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "c" * 40,
            "short_hash": "cccccccc",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        subject_shape_runs,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "shape-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
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


def test_render_subject_shape_overlay_from_mask_background(tmp_path: Path, monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    root = _build_refined_root(zarr_path)
    subject_shape_runs.write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run="refined_001",
        run_name="shape_001",
    )

    ctx = open_subject_shape_overlay_context(zarr_path, shape_run="shape_001", use_crop_images=False)
    fig = render_subject_shape_overlay(ctx, row=0, show_skeleton=True)

    assert fig.axes
    assert "Subject shape overlay" in fig.axes[0].get_title()
    labels = [collection.get_label() for collection in fig.axes[0].collections]
    assert "body skeleton" in labels
    assert "snout tip" in labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_reads_compact_mask_store(tmp_path: Path, monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    root = _build_refined_root(zarr_path)
    _replace_refined_masks_with_rle(root)
    subject_shape_runs.write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run="refined_001",
        run_name="shape_001",
    )

    ctx = open_subject_shape_overlay_context(zarr_path, shape_run="shape_001", use_crop_images=False)
    fig = render_subject_shape_overlay(ctx, row=0, show_skeleton=True)

    assert "masks_roi" not in ctx.refined_group
    assert ctx.mask_store.encoding == "component_rle_v1"
    labels = [collection.get_label() for collection in fig.axes[0].collections]
    assert "body skeleton" in labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_can_draw_persisted_eye_contours(tmp_path: Path, monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    root = _build_refined_root(zarr_path)
    _add_persisted_eye_left_contours(root)
    subject_shape_runs.write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run="refined_001",
        run_name="shape_001",
    )

    ctx = open_subject_shape_overlay_context(zarr_path, shape_run="shape_001", use_crop_images=False)
    fig = render_subject_shape_overlay(ctx, row=0, contour_source="persisted")

    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "eye left persisted" in labels
    assert "body contour" not in labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_can_offset_skeleton(tmp_path: Path, monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    root = _build_refined_root(zarr_path)
    subject_shape_runs.write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run="refined_001",
        run_name="shape_001",
    )

    ctx = open_subject_shape_overlay_context(zarr_path, shape_run="shape_001", use_crop_images=False)
    fig = render_subject_shape_overlay(ctx, row=0, show_skeleton=True, skeleton_style="offset")

    labels = [collection.get_label() for collection in fig.axes[0].collections]
    assert "body skeleton offset" in labels
    fig.canvas.draw()
    plt.close(fig)


def test_render_subject_shape_overlay_can_draw_spline_layers(tmp_path: Path, monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    root = _build_refined_root(zarr_path)
    subject_shape_runs.write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run="refined_001",
        run_name="shape_001",
    )
    body = root["analysis"]["subject_shape_runs"]["shape_001"]["components"]["subject_body"]
    body["bspline_sample_xy"][0, :, :] = np.nan
    body["bspline_sample_xy"][0, :4, :] = np.asarray(
        [[10.0, 4.0], [10.0, 8.0], [10.0, 12.0], [10.0, 16.0]],
        dtype=np.float32,
    )
    body["bspline_control_points_xy"][0, :, :] = np.nan
    body["bspline_control_points_xy"][0, :3, :] = np.asarray(
        [[10.0, 4.0], [11.0, 10.0], [10.0, 16.0]],
        dtype=np.float32,
    )
    body["tail_sample_xy"][0, :, :] = np.nan
    body["tail_sample_xy"][0, :3, :] = np.asarray(
        [[10.0, 10.0], [10.0, 13.0], [10.0, 16.0]],
        dtype=np.float32,
    )
    body["tail_normal_xy"][0, :, :] = np.nan
    body["tail_normal_xy"][0, :3, :] = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    body["bspline_valid"][0] = True
    body["tail_sample_valid"][0] = True

    ctx = open_subject_shape_overlay_context(zarr_path, shape_run="shape_001", use_crop_images=False)
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
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    root = _build_refined_root(zarr_path)
    refined = root["refined_subject_masks_runs"]["refined_001"]
    masks = np.asarray(refined["masks_roi"][:], dtype=np.uint8)
    masks[0, 0] = 0
    masks[0, 0, 3:17, 10] = 1
    masks[0, 0, 10, 10:18] = 1
    refined["masks_roi"][:] = masks
    subject_shape_runs.write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run="refined_001",
        run_name="shape_001",
    )

    ctx = open_subject_shape_overlay_context(zarr_path, shape_run="shape_001", use_crop_images=False)
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
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    root = _build_refined_root(zarr_path)
    _add_persisted_eye_left_contours(root)
    subject_shape_runs.write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run="refined_001",
        run_name="shape_001",
    )

    ctx = open_subject_shape_overlay_context(zarr_path, shape_run="shape_001", use_crop_images=False)
    fig = render_subject_shape_overlay(ctx, row=0, contour_source="compare")

    labels = [line.get_label() for line in fig.axes[0].lines]
    assert "eye left from mask" in labels
    assert "eye left persisted" in labels
    assert "body contour from mask" in labels
    fig.canvas.draw()
    plt.close(fig)


def test_export_subject_shape_overlay_png(tmp_path: Path, monkeypatch) -> None:
    _patch_provenance(monkeypatch)
    zarr_path = tmp_path / "shape.zarr"
    root = _build_refined_root(zarr_path)
    subject_shape_runs.write_subject_shape_run_group(
        root,
        zarr_path=zarr_path,
        refined_run="refined_001",
        run_name="shape_001",
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
