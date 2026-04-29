from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fisheye.analysis import subject_shape_runs
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
    fig = render_subject_shape_overlay(ctx, row=0)

    assert fig.axes
    assert "Subject shape overlay" in fig.axes[0].get_title()
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
    )

    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
