from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.utils.backfill_refined_subject_component_contours import (
    backfill_refined_subject_component_contours,
)


def _disk_mask(height: int, width: int, center_y: float, center_x: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2).astype(np.uint8)


def _build_refined_zarr(store_path: Path, *, include_swim_label: bool = True) -> zarr.Group:
    root = zarr.open_group(str(store_path), mode="w")
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_001"
    run = parent.create_group("refined_001")
    labels = ["subject_body", "eye_left", "eye_right"]
    if include_swim_label:
        labels.append("swim_bladder")
    run.attrs.update(
        {
            "mask_labels": labels,
            "label_schema_id": "subject_v1_lr",
        }
    )
    run.create_array("available_channels", data=np.ones((len(labels),), dtype=bool), overwrite=True)
    masks = np.zeros((2, len(labels), 24, 24), dtype=np.uint8)
    masks[0, 0, 4:20, 8:16] = 1
    masks[1, 0, 5:19, 7:17] = 1
    if include_swim_label:
        swim_idx = labels.index("swim_bladder")
        masks[0, swim_idx] = _disk_mask(24, 24, 14, 12, 3.0)
        masks[1, swim_idx] = _disk_mask(24, 24, 15, 11, 2.5)
    run.create_array("masks_roi", data=masks, chunks=(1, 1, 24, 24), overwrite=True)
    return root


def test_backfill_refined_subject_component_contours_dry_run_does_not_write(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _build_refined_zarr(zarr_path)

    summary = backfill_refined_subject_component_contours(
        zarr_path,
        components=["subject_body", "swim_bladder"],
        apply=False,
    )

    assert summary["would_write_count"] == 2
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_001"]
    assert "components" not in run

def test_backfill_refined_subject_component_contours_writes_body_and_swim(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _build_refined_zarr(zarr_path)

    summary = backfill_refined_subject_component_contours(
        zarr_path,
        components=["subject_body", "swim_bladder"],
        apply=True,
        chunk_size=1,
    )

    assert summary["written_count"] == 2
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_001"]
    for component in ("subject_body", "swim_bladder"):
        contours = run[f"components/{component}/contours"]
        assert contours.attrs["contour_schema_id"] == "component_contours_v1"
        assert contours.attrs["schema_id"] == "component_contours_v1"
        assert contours.attrs["method"] == "largest_external_contour"
        assert contours.attrs["boundary_policy"] == "external_only"
        assert contours.attrs["source_component"] == component
        assert contours.attrs["source_mask_run"] == "refined_001"
        assert tuple(contours["ptr"].shape) == (2,)
        assert tuple(contours["len"].shape) == (2,)
        assert contours["points_xy"].shape[1] == 2
        assert np.all(np.asarray(contours["ptr"][:]) >= 0)
        assert np.all(np.asarray(contours["len"][:]) > 0)


def test_backfill_refined_subject_component_contours_skips_existing_without_overwrite(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _build_refined_zarr(zarr_path)
    first = backfill_refined_subject_component_contours(
        zarr_path,
        components=["subject_body"],
        apply=True,
    )
    assert first["written_count"] == 1

    root = zarr.open_group(str(zarr_path), mode="a")
    contours = root["refined_subject_masks_runs/refined_001/components/subject_body/contours"]
    contours.attrs["sentinel"] = "keep"

    second = backfill_refined_subject_component_contours(
        zarr_path,
        components=["subject_body"],
        apply=True,
        overwrite=False,
    )

    assert second["existing_count"] == 1
    root = zarr.open_group(str(zarr_path), mode="r")
    contours = root["refined_subject_masks_runs/refined_001/components/subject_body/contours"]
    assert contours.attrs["sentinel"] == "keep"


def test_backfill_refined_subject_component_contours_missing_label_does_not_create_component(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    _build_refined_zarr(zarr_path, include_swim_label=False)

    summary = backfill_refined_subject_component_contours(
        zarr_path,
        components=["swim_bladder"],
        apply=True,
    )

    component_summary = summary["components"][0]
    assert component_summary["status"] == "missing_label"
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_001"]
    assert "components" not in run
