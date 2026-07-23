from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.subject_shape_io import (
    SubjectShapeIOError,
    discover_subject_shape_run_options,
    first_array_length,
    load_subject_body_component,
    load_subject_shape_run_tables,
    resolve_subject_shape_run,
)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values, chunks=values.shape, overwrite=True)


def _make_subject_shape_archive(tmp_path: Path) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "subject_shape.zarr"), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("subject_shape_runs")
    parent.attrs["latest"] = "shape_1"
    run = parent.create_group("shape_1")
    run.attrs.update(
        {
            "schema_id": "analysis.subject_shape_runs",
            "schema_version": 3,
            "method": "subject_shape_from_refined_masks_v8",
            "row_axis": "refined_subject_mask_rows",
            "component_names": ["subject_body", "eye_left", "eye_right"],
            "relation_names": ["eye_pair"],
            "source_refined_subject_masks_run": "refined_subject_1",
            "body_frame_schema_id": "fish_anatomical_body_frame",
        }
    )

    row_index = run.create_group("row_index")
    _write_array(row_index, "frame_indices", np.asarray([10, 11, 12], dtype=np.int64))
    _write_array(row_index, "detection_indices", np.asarray([0, 1, 2], dtype=np.int64))

    source = run.create_group("source_refined_subject_masks")
    source.attrs["source_run"] = "refined_subject_1"
    source.attrs["component_names"] = ["subject_body", "eye_left", "eye_right"]
    _write_array(source, "row_revision", np.asarray([[1, 2, 3], [1, 2, 4], [2, 2, 4]], dtype=np.int64))
    _write_array(source, "row_revision_available", np.asarray([True, True, True], dtype=bool))

    components = run.create_group("components")
    body = components.create_group("subject_body")
    body.attrs["component_name"] = "subject_body"
    body.attrs["tail_sample_count"] = 4
    _write_array(body, "mask_present", np.asarray([True, True, False], dtype=bool))
    _write_array(body, "area_px", np.asarray([100.0, 101.0, np.nan], dtype=np.float32))
    _write_array(body, "tail_sample_s", np.linspace(0.0, 1.0, 4, dtype=np.float32))
    _write_array(body, "tail_sample_xy", np.zeros((3, 4, 2), dtype=np.float32))
    _write_array(body, "tail_tangent_xy", np.ones((3, 4, 2), dtype=np.float32))
    _write_array(body, "tail_curvature_px_inv", np.zeros((3, 4), dtype=np.float32))
    _write_array(body, "tail_sample_valid", np.asarray([True, True, False], dtype=bool))
    _write_array(body, "bspline_valid", np.asarray([True, True, False], dtype=bool))
    _write_array(body, "tail_base_xy", np.zeros((3, 2), dtype=np.float32))
    _write_array(body, "head_endpoint_xy", np.ones((3, 2), dtype=np.float32))

    for component_name in ("eye_left", "eye_right"):
        eye = components.create_group(component_name)
        eye.attrs["component_name"] = component_name
        _write_array(eye, "ellipse_params", np.zeros((3, 5), dtype=np.float32))
        _write_array(eye, "ellipse_success", np.asarray([True, False, True], dtype=bool))

    body_frame = run.create_group("body_frame")
    body_frame.attrs["body_frame_schema_id"] = "fish_anatomical_body_frame"
    _write_array(body_frame, "origin_xy", np.asarray([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32))
    _write_array(body_frame, "forward_axis_xy", np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (3, 1)))
    _write_array(body_frame, "left_axis_xy", np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (3, 1)))
    _write_array(body_frame, "heading_deg", np.asarray([0.0, 1.0, 2.0], dtype=np.float32))
    _write_array(body_frame, "valid", np.asarray([True, True, False], dtype=bool))

    relations = run.create_group("relations")
    eye_pair = relations.create_group("eye_pair")
    eye_pair.attrs["relation_schema_id"] = "analysis.subject_shape.eye_pair_v1"
    _write_array(eye_pair, "separation_px", np.asarray([20.0, 21.0, np.nan], dtype=np.float32))
    _write_array(eye_pair, "separation_valid", np.asarray([True, True, False], dtype=bool))
    _write_array(eye_pair, "midpoint_xy", np.zeros((3, 2), dtype=np.float32))
    return root


def test_discover_subject_shape_run_options_uses_latest_and_shape_metadata(tmp_path: Path) -> None:
    root = _make_subject_shape_archive(tmp_path)

    options = discover_subject_shape_run_options(root, historical_inspection=True)

    assert len(options) == 1
    option = options[0]
    assert option.run_name == "shape_1"
    assert option.run_path == "analysis/subject_shape_runs/shape_1"
    assert option.schema_version == 3
    assert option.method == "subject_shape_from_refined_masks_v8"
    assert option.row_axis == "refined_subject_mask_rows"
    assert option.component_names == ("subject_body", "eye_left", "eye_right")
    assert option.relation_names == ("eye_pair",)
    assert option.n_rows == 3
    assert option.is_latest is True
    assert "latest" in option.label


def test_load_subject_shape_run_tables_reads_logical_groups(tmp_path: Path) -> None:
    root = _make_subject_shape_archive(tmp_path)

    tables = load_subject_shape_run_tables(
        root,
        run_name="analysis/subject_shape_runs/shape_1",
        historical_inspection=True,
    )

    assert tables.run_name == "shape_1"
    assert tables.run_path == "analysis/subject_shape_runs/shape_1"
    assert tables.schema_version == 3
    assert tables.row_axis == "refined_subject_mask_rows"
    assert tables.component_names == ("subject_body", "eye_left", "eye_right")
    assert tables.relation_names == ("eye_pair",)
    assert first_array_length(tables.require_component("subject_body").arrays) == 3
    assert tables.row_index["frame_indices"].tolist() == [10, 11, 12]
    assert tables.source_refined_subject_masks["row_revision_available"].tolist() == [True, True, True]
    np.testing.assert_allclose(tables.require_body_frame_array("origin_xy")[0], [1.0, 2.0])
    np.testing.assert_allclose(
        tables.relations["eye_pair"].arrays["separation_px"],
        [20.0, 21.0, np.nan],
        equal_nan=True,
    )
    assert (
        tables.source_paths["components/subject_body/tail_sample_xy"]
        == "analysis/subject_shape_runs/shape_1/components/subject_body/tail_sample_xy"
    )


def test_load_subject_body_component_can_limit_to_body_and_body_frame(tmp_path: Path) -> None:
    root = _make_subject_shape_archive(tmp_path)

    tables, body = load_subject_body_component(
        root,
        run_name="latest",
        include_body_frame=True,
        array_names=("tail_sample_xy", "tail_sample_valid"),
        historical_inspection=True,
    )

    assert tables.component_names == ("subject_body",)
    assert tables.relation_names == ()
    assert body.require_array("tail_sample_xy").shape == (3, 4, 2)
    assert set(body.arrays) == {"tail_sample_xy", "tail_sample_valid"}
    assert tables.require_body_frame_array("valid").tolist() == [True, True, False]


def test_resolve_subject_shape_run_rejects_missing_run(tmp_path: Path) -> None:
    root = _make_subject_shape_archive(tmp_path)

    with pytest.raises(SubjectShapeIOError, match="not found"):
        resolve_subject_shape_run(root, "missing_shape")


@pytest.mark.parametrize(
    "run_spec",
    (
        "garbage/shape_1",
        "analysis/subject_shape_runs/shape_1/extra",
    ),
)
def test_subject_shape_run_rejects_nonexact_explicit_path(
    tmp_path: Path,
    run_spec: str,
) -> None:
    root = _make_subject_shape_archive(tmp_path)

    with pytest.raises(SubjectShapeIOError, match="bare child name or the exact path"):
        resolve_subject_shape_run(root, run_spec, historical_inspection=True)


@pytest.mark.parametrize(
    ("latest", "latest_complete", "new_eligible"),
    (
        ("shape_1", "shape_2", True),
        ("shape_2", "shape_1", True),
        ("shape_2", "shape_2", False),
    ),
)
def test_subject_shape_latest_fails_closed_during_selector_handoff(
    tmp_path: Path,
    latest: str,
    latest_complete: str,
    new_eligible: bool,
) -> None:
    root = _make_subject_shape_archive(tmp_path)
    parent = root["analysis/subject_shape_runs"]
    old = parent["shape_1"]
    old.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    new = parent.create_group("shape_2")
    new.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": new_eligible,
        }
    )
    parent.attrs.update(
        {
            "latest": latest,
            "latest_complete": latest_complete,
        }
    )

    with pytest.raises(SubjectShapeIOError, match="No stable complete selector-eligible"):
        resolve_subject_shape_run(root, "latest")


def test_require_component_reports_available_components(tmp_path: Path) -> None:
    root = _make_subject_shape_archive(tmp_path)
    tables = load_subject_shape_run_tables(
        root,
        component_names=("subject_body",),
        relation_names=(),
        historical_inspection=True,
    )

    with pytest.raises(SubjectShapeIOError, match="available components: subject_body"):
        tables.require_component("swim_bladder")


def test_future_reader_rejects_roi_only_historical_run_without_explicit_inspection(
    tmp_path: Path,
) -> None:
    root = _make_subject_shape_archive(tmp_path)

    assert discover_subject_shape_run_options(root) == []
    with pytest.raises(
        SubjectShapeIOError,
        match="not a valid canonical coordinate publication",
    ):
        resolve_subject_shape_run(root, "shape_1")
    with pytest.raises(SubjectShapeIOError, match="not a valid canonical coordinate publication"):
        load_subject_shape_run_tables(root, run_name="shape_1")

    group, name, path = resolve_subject_shape_run(
        root,
        "shape_1",
        historical_inspection=True,
    )
    assert isinstance(group, zarr.Group)
    assert (name, path) == ("shape_1", "analysis/subject_shape_runs/shape_1")
