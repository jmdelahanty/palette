from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.analysis.eye_angle_io import (
    EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
    EyeAngleIOError,
    catalog_eye_angle_series,
    discover_eye_angle_run_options,
    first_array_length,
    load_eye_angle_run_tables,
    load_eye_angle_series_window,
    load_eye_gaze_frame_series,
    optional_1d_array,
    resolve_eye_angle_run,
)


class _SelectionGroup(dict[str, object]):
    def __init__(self, *, attrs: dict[str, object] | None = None) -> None:
        super().__init__()
        self.attrs = attrs if attrs is not None else {}

    def group_keys(self) -> list[str]:
        return [
            key
            for key, value in self.items()
            if isinstance(value, _SelectionGroup)
        ]


def test_explicit_run_resolution_rejects_descendant_and_alias_paths() -> None:
    parent = _SelectionGroup()
    root = _SelectionGroup()
    root["analysis/eye_angle_runs"] = parent

    for value in (
        "analysis/eye_angle_runs/run_1/support/frame_indices",
        "other/alias/run_1",
    ):
        with pytest.raises(EyeAngleIOError, match="descendant and alias paths"):
            resolve_eye_angle_run(root, run_name=value)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values, chunks=values.shape, overwrite=True)


def _fixed_text(values: list[str], width: int = 64) -> np.ndarray:
    out = np.zeros((len(values), width), dtype=np.uint8)
    for idx, value in enumerate(values):
        encoded = value.encode("utf-8")[: max(0, width - 1)]
        out[idx, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return out


def _make_eye_angle_archive(tmp_path: Path) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "eye_angle.zarr"), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("eye_angle_runs")
    parent.attrs["latest"] = "eye_angle_1"
    parent.attrs["latest_complete"] = "eye_angle_1"
    run = parent.create_group("eye_angle_1")
    run.attrs.update(
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 5,
            "preferred_angle_family": "gaze",
            "preferred_eye_axis": "ellipse_major",
            "row_axis": "keypoint_detection_rows",
            "fps": 100.0,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )
    angles = run.create_group("angles")
    roi = angles.create_group("roi")
    frame = angles.create_group("frame")
    qa = run.create_group("qa")
    qa_roi = qa.create_group("roi")
    qa_frame = qa.create_group("frame")
    support = run.create_group("support")

    _write_array(roi, "left_eye_angle_deg", np.asarray([10.0, 11.0, 12.0], dtype=np.float32))
    _write_array(roi, "left_gaze_signed_deg", np.asarray([-80.0, -79.0, -78.0], dtype=np.float32))
    _write_array(frame, "left_gaze_deg", np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    _write_array(frame, "right_gaze_deg", np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
    _write_array(frame, "vergence_gaze_deg", np.asarray([9.0, 10.0, 11.0, 12.0], dtype=np.float32))
    _write_array(frame, "vergence_gaze_signed_deg", np.asarray([-1.0, -2.0, -3.0, -4.0], dtype=np.float32))
    _write_array(qa_roi, "valid_frame", np.asarray([True, False, True], dtype=bool))
    _write_array(qa_frame, "valid_frame", np.asarray([True, False, True, True], dtype=bool))
    _write_array(support, "time_seconds", np.asarray([0.0, 0.01, 0.02], dtype=np.float64))
    _write_array(support, "frame_indices", np.asarray([0, 1, 2], dtype=np.int64))
    _write_array(support, "frame_time_seconds", np.asarray([0.0, 0.01, 0.02, 0.03], dtype=np.float64))
    return root


def _make_compact_eye_angle_archive(tmp_path: Path) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "eye_angle_compact.zarr"), mode="w")
    analysis = root.create_group("analysis")
    parent = analysis.create_group("eye_angle_runs")
    parent.attrs["latest"] = "eye_angle_compact"
    parent.attrs["latest_complete"] = "eye_angle_compact"
    run = parent.create_group("eye_angle_compact")
    run.attrs.update(
        {
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 5,
            "layout": EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2,
            "preferred_angle_family": "gaze",
            "preferred_eye_axis": "ellipse_major",
            "row_axis": "keypoint_detection_rows",
            "fps": 100.0,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        }
    )

    angle_names = _fixed_text(
        [
            "left_eye_angle_deg",
            "left_gaze_deg",
            "right_gaze_deg",
            "vergence_gaze_deg",
            "vergence_gaze_signed_deg",
        ]
    )
    angle_index = run.create_group("angle_channel_index")
    _write_array(angle_index, "name", angle_names)
    _write_array(
        angle_index,
        "representation",
        _fixed_text(["eye_frame", "gaze", "gaze", "gaze", "gaze"], width=32),
    )
    _write_array(
        run,
        "roi_angles",
        np.asarray(
            [
                [10.0, 1.0, 5.0, 9.0, -1.0],
                [11.0, 2.0, 6.0, 10.0, -2.0],
                [12.0, 3.0, 7.0, 11.0, -3.0],
            ],
            dtype=np.float32,
        ),
    )
    _write_array(
        run,
        "frame_angles",
        np.asarray(
            [
                [10.0, 1.0, 5.0, 9.0, -1.0],
                [11.0, 2.0, 6.0, 10.0, -2.0],
                [12.0, 3.0, 7.0, 11.0, -3.0],
                [13.0, 4.0, 8.0, 12.0, -4.0],
            ],
            dtype=np.float32,
        ),
    )

    vector_index = run.create_group("vector_channel_index")
    _write_array(vector_index, "name", _fixed_text(["left_gaze_xy", "right_gaze_xy"], width=32))
    _write_array(
        run,
        "roi_vectors",
        np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.9, 0.1], [0.1, 0.9]],
                [[0.8, 0.2], [0.2, 0.8]],
            ],
            dtype=np.float32,
        ),
    )

    qa_index = run.create_group("qa_channel_index")
    _write_array(qa_index, "name", _fixed_text(["valid_frame", "major_axis_marginal"], width=32))
    _write_array(
        run,
        "roi_qa",
        np.asarray([[True, False], [False, True], [True, False]], dtype=bool),
    )
    _write_array(
        run,
        "frame_qa",
        np.asarray([[True, False], [False, True], [True, False], [True, False]], dtype=bool),
    )

    support = run.create_group("support")
    _write_array(support, "time_seconds", np.asarray([0.0, 0.01, 0.02], dtype=np.float64))
    _write_array(support, "frame_indices", np.asarray([0, 1, 2], dtype=np.int64))
    _write_array(support, "frame_time_seconds", np.asarray([0.0, 0.01, 0.02, 0.03], dtype=np.float64))
    return root


def test_discover_eye_angle_run_options_uses_latest_and_shape_metadata(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)

    assert discover_eye_angle_run_options(root) == []
    options = discover_eye_angle_run_options(root, legacy_compatibility=True)

    assert len(options) == 1
    option = options[0]
    assert option.run_name == "eye_angle_1"
    assert option.run_path == "analysis/eye_angle_runs/eye_angle_1"
    assert option.schema_version == 5
    assert option.preferred_angle_family == "gaze"
    assert option.preferred_eye_axis == "ellipse_major"
    assert option.n_rows == 3
    assert option.is_latest is True
    assert "latest" in option.label


def _make_eye_angle_selection_root() -> _SelectionGroup:
    root = _SelectionGroup()
    parent = _SelectionGroup(
        attrs={"latest": "eye_angle_1", "latest_complete": "eye_angle_1"}
    )
    root["analysis/eye_angle_runs"] = parent
    parent["eye_angle_1"] = _SelectionGroup(
        attrs={
            "schema_id": "analysis.eye_angle_runs",
            "schema_version": 7,
            "layout": "compact_dense_v2",
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": True,
        },
    )
    return root


def test_eye_angle_reader_rejects_explicit_ineligible_run() -> None:
    root = _make_eye_angle_selection_root()
    root["analysis/eye_angle_runs"]["eye_angle_1"].attrs[
        "stage_selector_eligible"
    ] = False

    with pytest.raises(EyeAngleIOError, match="not selector-eligible"):
        resolve_eye_angle_run(root, "eye_angle_1")

    assert discover_eye_angle_run_options(root) == []


def test_eye_angle_reader_fails_closed_during_selector_activation(
) -> None:
    root = _make_eye_angle_selection_root()
    parent = root["analysis/eye_angle_runs"]
    parent.attrs["latest"] = "candidate"
    parent.attrs["latest_complete"] = "eye_angle_1"
    parent["candidate"] = _SelectionGroup(
        attrs={
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        },
    )

    with pytest.raises(EyeAngleIOError, match="selector activation may be in progress"):
        resolve_eye_angle_run(root)


def test_eye_angle_legacy_run_requires_explicit_compatibility(
) -> None:
    root = _make_eye_angle_selection_root()
    parent = root["analysis/eye_angle_runs"]
    run = parent["eye_angle_1"]
    del parent.attrs["latest_complete"]
    del run.attrs["palette_run_completion_status"]
    del run.attrs["stage_selector_eligible"]

    with pytest.raises(EyeAngleIOError, match="No stable complete"):
        resolve_eye_angle_run(root)

    resolved, resolved_name, _path = resolve_eye_angle_run(
        root,
        legacy_compatibility=True,
    )
    assert resolved is run
    assert resolved_name == "eye_angle_1"


@pytest.mark.parametrize(
    "requested_run",
    ("  eye_angle_1  ", "/analysis/eye_angle_runs/eye_angle_1/"),
)
def test_eye_angle_resolution_normalizes_explicit_run(
    requested_run: str,
) -> None:
    root = _make_eye_angle_selection_root()

    _run, resolved_name, _path = resolve_eye_angle_run(root, requested_run)

    assert resolved_name == "eye_angle_1"


def test_load_eye_angle_run_tables_reads_logical_groups(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)

    with pytest.raises(EyeAngleIOError, match="legacy_compatibility=True"):
        load_eye_angle_run_tables(
            root,
            run_name="analysis/eye_angle_runs/eye_angle_1",
        )
    tables = load_eye_angle_run_tables(
        root,
        run_name="analysis/eye_angle_runs/eye_angle_1",
        legacy_compatibility=True,
    )

    assert tables.run_name == "eye_angle_1"
    assert tables.run_path == "analysis/eye_angle_runs/eye_angle_1"
    assert tables.schema_version == 5
    assert tables.row_axis == "keypoint_detection_rows"
    assert first_array_length(tables.roi) == 3
    assert optional_1d_array(tables.support, "frame_time_seconds", length=4) is not None
    np.testing.assert_allclose(tables.roi["left_eye_angle_deg"], [10.0, 11.0, 12.0])
    assert tables.qa_frame["valid_frame"].tolist() == [True, False, True, True]


def test_load_eye_angle_run_tables_reads_compact_dense_channels(tmp_path: Path) -> None:
    root = _make_compact_eye_angle_archive(tmp_path)

    assert discover_eye_angle_run_options(root) == []
    options = discover_eye_angle_run_options(root, legacy_compatibility=True)
    assert len(options) == 1
    assert options[0].run_name == "eye_angle_compact"
    assert options[0].n_rows == 3

    tables = load_eye_angle_run_tables(
        root,
        run_name="latest",
        legacy_compatibility=True,
    )

    assert tables.attrs["layout"] == EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2
    assert first_array_length(tables.roi) == 3
    np.testing.assert_allclose(tables.roi["left_eye_angle_deg"], [10.0, 11.0, 12.0])
    np.testing.assert_allclose(tables.frame["left_gaze_deg"], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(tables.roi["left_gaze_xy"], [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    assert tables.qa_frame["valid_frame"].tolist() == [True, False, True, True]
    assert tables.source_paths[
        "analysis/eye_angle_runs/eye_angle_compact/angles/frame/left_gaze_deg"
    ].endswith("frame_angles[:,1]")


def test_compact_eye_angle_window_reads_selected_columns_and_rows(tmp_path: Path) -> None:
    root = _make_compact_eye_angle_archive(tmp_path)

    catalog = catalog_eye_angle_series(
        root,
        run_name="latest",
        prefer_frame=True,
        legacy_compatibility=True,
    )

    assert catalog.row_axis == "frame"
    assert catalog.row_count == 4
    assert catalog.time_start_s == 0.0
    assert catalog.time_stop_s == 0.03
    assert catalog.channel_representations["left_gaze_deg"] == "gaze"
    assert catalog.qa_channels == ("valid_frame", "major_axis_marginal")

    window = load_eye_angle_series_window(
        root,
        run_name="latest",
        start_s=0.01,
        stop_s=0.02,
        angle_channels=("left_gaze_deg", "vergence_gaze_signed_deg"),
        legacy_compatibility=True,
    )

    np.testing.assert_allclose(window.time_seconds, [0.01, 0.02])
    assert window.frame_indices.tolist() == [1, 2]
    np.testing.assert_allclose(window.angles["left_gaze_deg"], [2.0, 3.0])
    np.testing.assert_allclose(window.angles["vergence_gaze_signed_deg"], [-2.0, -3.0])
    assert window.qa["valid_frame"].tolist() == [False, True]
    assert window.qa["major_axis_marginal"].tolist() == [True, False]


def test_eye_angle_window_refuses_unbounded_large_projection(tmp_path: Path) -> None:
    root = _make_compact_eye_angle_archive(tmp_path)

    with pytest.raises(EyeAngleIOError, match="viewer limit"):
        load_eye_angle_series_window(
            root,
            run_name="latest",
            angle_channels=("left_gaze_deg",),
            max_rows=2,
            legacy_compatibility=True,
        )


def test_load_eye_gaze_frame_series_aligns_frames_and_validity(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)
    frames = np.asarray([0, 2, 3], dtype=np.int64)

    series, source_refs = load_eye_gaze_frame_series(
        root,
        eye_angle_run="latest",
        eye_angle_family="gaze",
        frames=frames,
        allowed_families=("gaze",),
        legacy_compatibility=True,
    )

    np.testing.assert_allclose(series["left_gaze_deg"], [1.0, 3.0, 4.0])
    np.testing.assert_allclose(series["right_gaze_deg"], [5.0, 7.0, 8.0])
    np.testing.assert_allclose(series["vergence_gaze_signed_deg"], [-1.0, -3.0, -4.0])
    assert series["valid_frame"].tolist() == [True, True, True]
    assert source_refs["source_eye_angle_run"] == "eye_angle_1"
    assert source_refs["source_eye_angle_schema_version"] == 5
    assert source_refs["source_eye_angle_arrays"]["left_gaze_deg"].endswith(
        "/angles/frame/left_gaze_deg"
    )


def test_load_eye_gaze_frame_series_uses_compact_source_paths(tmp_path: Path) -> None:
    root = _make_compact_eye_angle_archive(tmp_path)
    frames = np.asarray([0, 2, 3], dtype=np.int64)

    series, source_refs = load_eye_gaze_frame_series(
        root,
        eye_angle_run="latest",
        eye_angle_family="gaze",
        frames=frames,
        allowed_families=("gaze",),
        legacy_compatibility=True,
    )

    np.testing.assert_allclose(series["left_gaze_deg"], [1.0, 3.0, 4.0])
    np.testing.assert_allclose(series["right_gaze_deg"], [5.0, 7.0, 8.0])
    np.testing.assert_allclose(series["vergence_gaze_deg"], [9.0, 11.0, 12.0])
    assert series["valid_frame"].tolist() == [True, True, True]
    assert source_refs["source_eye_angle_run"] == "eye_angle_compact"
    assert source_refs["source_eye_angle_arrays"]["left_gaze_deg"].endswith("frame_angles[:,1]")
    assert source_refs["source_eye_angle_arrays"]["valid_frame"].endswith("frame_qa[:,0]")


def test_load_eye_gaze_frame_series_rejects_unsupported_family(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)

    with pytest.raises(EyeAngleIOError, match="Unsupported eye_angle_family"):
        load_eye_gaze_frame_series(
            root,
            eye_angle_run="latest",
            eye_angle_family="eye_frame",
            frames=np.asarray([0], dtype=np.int64),
            allowed_families=("gaze",),
            legacy_compatibility=True,
        )


def test_load_eye_gaze_frame_series_checks_frame_bounds(tmp_path: Path) -> None:
    root = _make_eye_angle_archive(tmp_path)

    with pytest.raises(EyeAngleIOError, match="cannot cover requested frame"):
        load_eye_gaze_frame_series(
            root,
            eye_angle_run="latest",
            eye_angle_family="gaze",
            frames=np.asarray([4], dtype=np.int64),
            allowed_families=("gaze",),
            legacy_compatibility=True,
        )
