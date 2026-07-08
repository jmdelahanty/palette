import numpy as np
import zarr

from fisheye.analysis.track_kinematics import (
    DetectionResolution,
    _crop_row_source_label,
    _filter_public_track_rows,
    _ordered_track_arena_ids,
    load_offline_position_source,
)


def test_filter_public_track_rows_excludes_unassigned_by_default() -> None:
    detection_source = np.array([0, 1, 0, 1], dtype=np.int8)

    track_ids, frames, positions_px, headings_deg, keypoint_success, filtered_source = (
        _filter_public_track_rows(
            track_ids=np.array([-1, 0, 1, -1], dtype=np.int64),
            frames=np.array([10, 11, 12, 13], dtype=np.int64),
            positions_px=np.array(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                dtype=np.float64,
            ),
            headings_deg=np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64),
            keypoint_success=np.array([True, False, True, False], dtype=bool),
            detection_source=detection_source,
            include_unassigned=False,
        )
    )

    assert track_ids.tolist() == [0, 1]
    assert frames.tolist() == [11, 12]
    assert positions_px.tolist() == [[2.0, 2.0], [3.0, 3.0]]
    assert headings_deg.tolist() == [10.0, 20.0]
    assert keypoint_success.tolist() == [False, True]
    assert filtered_source is not None
    assert filtered_source.tolist() == [1, 0]


def test_filter_public_track_rows_keeps_unassigned_when_requested() -> None:
    track_ids, frames, positions_px, headings_deg, keypoint_success, filtered_source = (
        _filter_public_track_rows(
            track_ids=np.array([-1, 0], dtype=np.int64),
            frames=np.array([5, 6], dtype=np.int64),
            positions_px=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
            headings_deg=np.array([15.0, 30.0], dtype=np.float64),
            keypoint_success=np.array([True, True], dtype=bool),
            detection_source=None,
            include_unassigned=True,
        )
    )

    assert track_ids.tolist() == [-1, 0]
    assert frames.tolist() == [5, 6]
    assert positions_px.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert headings_deg.tolist() == [15.0, 30.0]
    assert keypoint_success.tolist() == [True, True]
    assert filtered_source is None


def test_ordered_track_arena_ids_allows_diagnostic_unassigned_track() -> None:
    result = _ordered_track_arena_ids([-1, 0, 1], {0: 5, 1: 9})

    assert result is not None
    assert result.tolist() == [-1, 5, 9]


def test_load_offline_position_source_prefers_crop_rows() -> None:
    root = zarr.group()
    crop_group = root.create_group("crop")
    crop_group.create_array(
        "bbox_norm_coords",
        data=np.zeros((3, 4), dtype=np.float32),
        chunks=(3, 4),
        overwrite=True,
    )
    crop_group.create_array(
        "frame_indices",
        data=np.array([10, 11, 12], dtype=np.int64),
        chunks=(3,),
        overwrite=True,
    )
    crop_group.create_array(
        "detection_source",
        data=np.array([1, 1, 0], dtype=np.int8),
        chunks=(3,),
        overwrite=True,
    )

    detection_group = root.create_group("refined_instances")
    detection_group.create_array(
        "bbox_norm_coords",
        data=np.zeros((2, 4), dtype=np.float32),
        chunks=(2, 4),
        overwrite=True,
    )
    detection_group.create_array(
        "frame_indices",
        data=np.array([10, 12], dtype=np.int64),
        chunks=(2,),
        overwrite=True,
    )
    detection = DetectionResolution(
        group=detection_group,
        path="refined_detect_runs/refined_1/instances",
        is_refined=True,
        run_name="refined_1",
        variant="instances",
        source_detect_run="detect_1",
        parent_path="refined_detect_runs/refined_1",
    )

    source = load_offline_position_source(
        crop_group,
        crop_run_name="crop_1",
        detection=detection,
    )

    assert source.kind == "crop_rows"
    assert source.path == "crop_runs/crop_1"
    assert source.bbox_norm_coords.shape == (3, 4)
    assert source.frame_indices.tolist() == [10, 11, 12]
    assert source.detection_source is not None
    assert source.detection_source.tolist() == [1, 1, 0]


def test_load_offline_position_source_accepts_external_crop_rows_without_detection() -> None:
    root = zarr.group()
    crop_group = root.create_group("crop")
    crop_group.attrs["detection_source_type"] = (
        "external_crop_recorder_crop_meta_selected_live_detection"
    )
    crop_group.create_array(
        "bbox_norm_coords",
        data=np.ones((2, 4), dtype=np.float32),
        chunks=(2, 4),
        overwrite=True,
    )
    crop_group.create_array(
        "frame_indices",
        data=np.array([20, 21], dtype=np.int64),
        chunks=(2,),
        overwrite=True,
    )

    source = load_offline_position_source(
        crop_group,
        crop_run_name="crop_external",
        detection=None,
    )

    assert _crop_row_source_label(crop_group.attrs) == (
        "external_crop_recorder_crop_meta_selected_live_detection"
    )
    assert source.kind == "crop_rows"
    assert source.path == "crop_runs/crop_external"
    assert source.bbox_norm_coords.shape == (2, 4)
    assert source.frame_indices.tolist() == [20, 21]
    assert source.detection_source is None


def test_crop_row_source_label_accepts_clipped_collection_proxy() -> None:
    assert (
        _crop_row_source_label(
            {
                "detection_source_type": "finalized_clipped_refined_detect_collection_proxy",
            }
        )
        == "finalized_clipped_refined_detect_collection_proxy"
    )


def test_load_offline_position_source_rejects_crop_row_count_mismatch() -> None:
    root = zarr.group()
    crop_group = root.create_group("crop")
    crop_group.create_array(
        "bbox_norm_coords",
        data=np.ones((2, 4), dtype=np.float32),
        chunks=(2, 4),
        overwrite=True,
    )
    crop_group.create_array(
        "frame_indices",
        data=np.array([20, 21, 22], dtype=np.int64),
        chunks=(3,),
        overwrite=True,
    )

    try:
        load_offline_position_source(
            crop_group,
            crop_run_name="crop_bad",
            detection=None,
        )
    except ValueError as exc:
        assert "row count mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected row count mismatch to fail")
