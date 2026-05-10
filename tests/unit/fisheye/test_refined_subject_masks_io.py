from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.refined_subject_masks_io import (
    RefinedSubjectMasksIOError,
    discover_refined_subject_masks_run_options,
    load_refined_subject_masks_run_tables,
    resolve_refined_subject_masks_run,
)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values, chunks=values.shape, overwrite=True)


def _make_refined_subject_archive(tmp_path: Path) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "refined_subject.zarr"), mode="w")
    parent = root.create_group("refined_subject_masks_runs")
    parent.attrs["latest"] = "refined_1"
    run = parent.create_group("refined_1")
    run.attrs.update(
        {
            "method": "smart_finalizer",
            "label_schema_id": "subject_v1_lr",
            "mask_labels": ["subject_body", "eye_left", "eye_right", "swim_bladder"],
            "component_review_statuses": {
                "subject_body": {"state": "approved"},
                "eye_left": {"state": "approved"},
                "eye_right": {"state": "approved"},
                "swim_bladder": {"state": "missing"},
            },
        }
    )

    masks = np.zeros((3, 4, 8, 8), dtype=np.uint8)
    masks[:, 0, 2:6, 2:6] = 1
    masks[:, 1, 2:4, 1:3] = 1
    masks[:, 2, 2:4, 5:7] = 1
    _write_array(run, "masks_roi", masks)
    _write_array(run, "available_channels", np.asarray([True, True, True, False], dtype=bool))
    _write_array(run, "edit_applied", np.asarray([[False, False, False, False]] * 3, dtype=bool))
    _write_array(run, "detection_source", np.asarray([0, 0, 0], dtype=np.int8))
    _write_array(run, "frame_indices", np.asarray([10, 11, 12], dtype=np.int64))
    _write_array(run, "detection_indices", np.asarray([0, 1, 2], dtype=np.int64))

    metrics = run.create_group("metrics")
    _write_array(metrics, "mask_present", np.asarray([[True, True, True, False]] * 3, dtype=bool))
    _write_array(metrics, "area_px", np.asarray([[16.0, 4.0, 4.0, np.nan]] * 3, dtype=np.float32))
    _write_array(metrics, "centroid_xy", np.zeros((3, 4, 2), dtype=np.float32))
    _write_array(metrics, "centroid_valid", np.asarray([[True, True, True, False]] * 3, dtype=bool))
    _write_array(metrics, "bbox_xyxy", np.zeros((3, 4, 4), dtype=np.float32))
    _write_array(metrics, "bbox_valid", np.asarray([[True, True, True, False]] * 3, dtype=bool))

    components = run.create_group("components")
    body = components.create_group("subject_body")
    body.attrs["component_schema_id"] = "subject_body_v1"
    _write_array(body, "row_revision", np.asarray([0, 1, 0], dtype=np.int64))
    _write_array(body, "manual_override", np.asarray([False, True, False], dtype=bool))
    body_metrics = body.create_group("metrics")
    _write_array(body_metrics, "component_count", np.asarray([1, 1, 0], dtype=np.int32))
    _write_array(body_metrics, "largest_component_fraction", np.asarray([1.0, 1.0, np.nan], dtype=np.float32))
    qc = body.create_group("qc")
    _write_array(qc, "severe_qc_failure", np.asarray([False, True, False], dtype=bool))
    _write_array(qc, "requires_review", np.asarray([False, True, False], dtype=bool))

    for component_name in ("eye_left", "eye_right"):
        eye = components.create_group(component_name)
        _write_array(eye, "row_revision", np.asarray([0, 0, 0], dtype=np.int64))
        geometry = eye.create_group("geometry")
        _write_array(geometry, "ellipse_params", np.zeros((3, 5), dtype=np.float32))
        _write_array(geometry, "ellipse_success", np.asarray([True, True, False], dtype=bool))

    relations = run.create_group("relations")
    eye_pair = relations.create_group("eye_pair")
    eye_pair.attrs["relation_schema_id"] = "refined_subject_eye_pair_relation_v1"
    eye_pair_metrics = eye_pair.create_group("metrics")
    _write_array(eye_pair_metrics, "separation_px", np.asarray([20.0, 21.0, np.nan], dtype=np.float32))
    _write_array(eye_pair_metrics, "separation_valid", np.asarray([True, True, False], dtype=bool))
    return root


def test_discover_refined_subject_masks_options_uses_latest_and_mask_metadata(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    options = discover_refined_subject_masks_run_options(root)

    assert len(options) == 1
    option = options[0]
    assert option.run_name == "refined_1"
    assert option.run_path == "refined_subject_masks_runs/refined_1"
    assert option.method == "smart_finalizer"
    assert option.label_schema_id == "subject_v1_lr"
    assert option.mask_labels == ("subject_body", "eye_left", "eye_right", "swim_bladder")
    assert option.available_components == ("subject_body", "eye_left", "eye_right")
    assert option.n_rows == 3
    assert option.channel_count == 4
    assert option.roi_shape == (8, 8)
    assert option.is_latest is True
    assert "latest" in option.label


def test_load_refined_subject_masks_run_tables_reads_logical_surfaces(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    tables = load_refined_subject_masks_run_tables(root, run_name="refined_subject_masks_runs/refined_1")

    assert tables.run_name == "refined_1"
    assert tables.run_path == "refined_subject_masks_runs/refined_1"
    assert tables.mask_labels == ("subject_body", "eye_left", "eye_right", "swim_bladder")
    assert tables.component_index("eye_right") == 2
    assert tables.component_available("swim_bladder") is False
    assert tables.resolve_components(("subject_body", "swim_bladder")) == (("subject_body", 0),)
    assert tables.require_masks_roi().shape == (3, 4, 8, 8)
    assert tables.run_arrays["frame_indices"].tolist() == [10, 11, 12]
    assert tables.metrics["mask_present"].shape == (3, 4)
    assert tables.components["subject_body"].arrays["row_revision"].tolist() == [0, 1, 0]
    assert tables.components["subject_body"].qc["requires_review"].tolist() == [False, True, False]
    assert tables.components["eye_left"].geometry["ellipse_params"].shape == (3, 5)
    np.testing.assert_allclose(
        tables.relations["eye_pair"].metrics["separation_px"],
        [20.0, 21.0, np.nan],
        equal_nan=True,
    )
    assert tables.source_paths["masks_roi"] == "refined_subject_masks_runs/refined_1/masks_roi"
    assert (
        tables.source_paths["components/subject_body/qc/requires_review"]
        == "refined_subject_masks_runs/refined_1/components/subject_body/qc/requires_review"
    )


def test_load_refined_subject_masks_run_tables_can_skip_dense_masks(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    tables = load_refined_subject_masks_run_tables(root, include_masks_roi=False)

    assert tables.masks_roi is None
    assert tables.n_rows == 3
    with pytest.raises(RefinedSubjectMasksIOError, match="loaded without masks_roi"):
        tables.require_masks_roi()


def test_load_refined_subject_masks_run_tables_rejects_missing_component(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    with pytest.raises(RefinedSubjectMasksIOError, match="not present"):
        load_refined_subject_masks_run_tables(root, component_names=("missing_component",))


def test_resolve_refined_subject_masks_run_rejects_missing_run(tmp_path: Path) -> None:
    root = _make_refined_subject_archive(tmp_path)

    with pytest.raises(RefinedSubjectMasksIOError, match="not found"):
        resolve_refined_subject_masks_run(root, "missing_refined")
