from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.analysis.detection_occupancy_runs import (
    IMAGE_QUADRANTS_ZONE_SET_ID,
    SESSION_VISUALIZATION_CONTRACT_ID,
    OccupancyWindow,
    _resolve_epoch_run,
    build_detection_occupancy_result,
    build_session_occupancy_result,
    write_detection_occupancy_run,
    write_session_occupancy_run,
)
from fisheye.analysis.detection_occupancy_schema import (
    build_occupancy_array_declarations,
    validate_occupancy_array_manifest,
)
from fisheye.analysis_workflows.materializers.exact_tabular_candidate import (
    materialize_exact_tabular_candidate,
)


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=np.asarray(values), chunks=np.asarray(values).shape, overwrite=True)


def _bbox_from_centers(centers: list[tuple[float, float]]) -> np.ndarray:
    rows = []
    for x, y in centers:
        rows.append([x - 1.0, y - 1.0, x + 1.0, y + 1.0])
    return np.asarray(rows, dtype=np.float32)


def _make_detection_archive(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "occupancy_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", use_consolidated=False)
    root.attrs.update(
        {
            "recording_id": "quadrant_test",
            "width": 100,
            "height": 80,
            "fps": 10.0,
            "total_frames": 8,
        }
    )
    instances = root.require_group("refined_detect_runs").require_group("refined_1").require_group("instances")
    frames = np.asarray([0, 1, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    centers = [
        (10.0, 10.0),  # top_left
        (20.0, 10.0),  # duplicate frame 1, lower score
        (75.0, 10.0),  # duplicate frame 1, selected as top_right
        (10.0, 60.0),  # bottom_left
        (75.0, 60.0),  # bottom_right
        (50.0, 40.0),  # midlines go bottom_right
        (50.0, 10.0),  # x midline goes top_right
        (10.0, 40.0),  # y midline goes bottom_left
    ]
    _write_array(instances, "frame_indices", frames)
    _write_array(instances, "bbox_img_xyxy", _bbox_from_centers(centers))
    _write_array(instances, "confidence_scores", np.asarray([0.9, 0.1, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9]))
    return zarr_path


def test_latest_epoch_selector_resolves_authoritative_completed_run(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(
        str(tmp_path / "epochs.zarr"),
        mode="w",
        use_consolidated=False,
    )
    parent = root.require_group("analysis").require_group("stimulus_epoch_runs")
    parent.create_group("epochs_complete")
    parent.attrs["latest"] = "stale_epoch"
    parent.attrs["latest_complete"] = "epochs_complete"

    group, run_name, run_path = _resolve_epoch_run(root, "latest")

    assert group.name.endswith("/epochs_complete")
    assert run_name == "epochs_complete"
    assert run_path == "analysis/stimulus_epoch_runs/epochs_complete"


def test_detection_occupancy_writes_image_quadrant_spatial_summary(tmp_path: Path) -> None:
    zarr_path = _make_detection_archive(tmp_path)
    windows = (
        OccupancyWindow(0, "pre_event", 0, 3, 0.0, 0.4, 0.4),
        OccupancyWindow(1, "post_event", 4, 7, 0.4, 0.8, 0.4),
    )

    result = build_detection_occupancy_result(
        zarr_path,
        run_name="occupancy_1",
        stimulus_epoch_run="epochs_1",
        epoch_windows=windows,
        detection_path="refined_detect_runs/refined_1/instances",
        bin_size=50,
        smooth_sigma=0.0,
    )

    assert [zone_set.zone_set_id for zone_set in result.spatial_occupancy] == [IMAGE_QUADRANTS_ZONE_SET_ID]
    quadrants = result.spatial_occupancy[0]
    assert quadrants.zone_id == ("top_left", "top_right", "bottom_left", "bottom_right")
    np.testing.assert_array_equal(
        quadrants.frame_count,
        np.asarray(
            [
                [1, 1, 1, 1],
                [0, 1, 1, 1],
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(quadrants.detected_frame_count, np.asarray([4, 3], dtype=np.int64))
    np.testing.assert_array_equal(quadrants.missing_frame_count, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_allclose(quadrants.time_s, np.asarray([[0.1, 0.1, 0.1, 0.1], [0.0, 0.1, 0.1, 0.1]]))
    np.testing.assert_allclose(
        quadrants.fraction_of_epoch,
        np.asarray([[0.25, 0.25, 0.25, 0.25], [0.0, 0.25, 0.25, 0.25]]),
    )
    np.testing.assert_allclose(
        quadrants.fraction_of_detected,
        np.asarray([[0.25, 0.25, 0.25, 0.25], [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]),
    )

    run_path = write_detection_occupancy_run(zarr_path, result, overwrite=True, write_png=False)

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root[run_path]
    zone_group = run["spatial_occupancy"][IMAGE_QUADRANTS_ZONE_SET_ID]
    assert zone_group.attrs["schema_id"] == "palette.spatial_occupancy_zones.v1"
    assert zone_group.attrs["zone_set_source"] == "predefined_spec:quadrants.v1"
    stored_summary = zone_group["summary"]
    np.testing.assert_array_equal(stored_summary["frame_count"][:], quadrants.frame_count)
    np.testing.assert_array_equal(stored_summary["missing_frame_count"][:], quadrants.missing_frame_count)
    assert zone_group["zone_spec"]["bounds_xyxy"].shape == (4, 4)
    assert validate_occupancy_array_manifest(run, session=False) == ()
    declarations = build_occupancy_array_declarations(run, session=False)
    assert len(declarations) == 30
    assert all(declaration.byte_planner_adopted is False for declaration in declarations)


def test_session_occupancy_does_not_require_stimulus_epochs(tmp_path: Path) -> None:
    zarr_path = _make_detection_archive(tmp_path)

    result = build_session_occupancy_result(
        zarr_path,
        run_name="session_1",
        detection_path="refined_detect_runs/refined_1/instances",
        bin_size=50,
        smooth_sigma=0.0,
    )

    assert [(window.label, window.start_frame, window.end_frame) for window in result.windows] == [
        ("full_session", 0, 7)
    ]
    assert result.source_stimulus_epoch_path == "recording/full_session"

    run_path = write_session_occupancy_run(
        zarr_path,
        result,
        overwrite=True,
        write_png=True,
    )

    assert run_path == "analysis/session_occupancy_runs/session_1"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    run = root[run_path]
    assert run.attrs["schema_id"] == "palette.session_occupancy.v1"
    assert run.attrs["row_axis"] == "session_segments"
    assert run.attrs["source_segment_kind"] == "full_session"
    assert "source_stimulus_epoch_run" not in run.attrs
    artifact = run["visualizations/session_occupancy_overview_png"]
    assert artifact.attrs["visualization_contract_id"] == SESSION_VISUALIZATION_CONTRACT_ID
    assert artifact.attrs["renderer"] == "fisheye.analysis.detection_occupancy_runs"
    assert validate_occupancy_array_manifest(run, session=True) == ()
    assert len(build_occupancy_array_declarations(run, session=True)) == 29


def test_occupancy_contract_rejects_cross_family_lineage_inventory(
    tmp_path: Path,
) -> None:
    zarr_path = _make_detection_archive(tmp_path)
    result = build_session_occupancy_result(
        zarr_path,
        run_name="session_1",
        detection_path="refined_detect_runs/refined_1/instances",
        bin_size=50,
        smooth_sigma=0.0,
    )
    run_path = write_session_occupancy_run(
        zarr_path,
        result,
        overwrite=True,
        write_png=False,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    run = root[run_path]
    run["windows"].create_array(
        "source_stimulus_epoch_window_id",
        data=np.asarray([0], dtype=np.int32),
        chunks=(1,),
    )

    errors = validate_occupancy_array_manifest(run, session=True)

    assert any("Unexpected compact arrays" in error for error in errors)


def test_detection_occupancy_can_be_rematerialized_as_ineligible_byte_candidate(
    tmp_path: Path,
) -> None:
    zarr_path = _make_detection_archive(tmp_path)
    result = build_detection_occupancy_result(
        zarr_path,
        run_name="occupancy_source",
        stimulus_epoch_run="epochs_1",
        epoch_windows=(
            OccupancyWindow(0, "full", 0, 7, 0.0, 0.8, 0.8),
        ),
        detection_path="refined_detect_runs/refined_1/instances",
        bin_size=50,
        smooth_sigma=0.0,
    )
    write_detection_occupancy_run(
        zarr_path,
        result,
        overwrite=True,
        write_png=False,
    )

    published = materialize_exact_tabular_candidate(
        zarr_path,
        family_id="detection_occupancy",
        source_run="occupancy_source",
        run_name="occupancy_candidate",
        scratch_root=tmp_path / "scratch",
        copy_backend="python",
        apply=True,
    )

    assert published["status"] == "complete"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent = root["analysis/detection_occupancy_runs"]
    assert parent.attrs["latest_complete"] == "occupancy_source"
    candidate = parent["occupancy_candidate"]
    assert candidate.attrs["stage_selector_eligible"] is False
    assert candidate.attrs["storage_candidate_profile_promoted"] is False
    assert validate_occupancy_array_manifest(
        candidate,
        session=False,
        byte_planner_adopted=True,
    ) == ()
