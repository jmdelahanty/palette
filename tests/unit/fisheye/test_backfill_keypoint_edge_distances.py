import numpy as np
import zarr

from fisheye.utils.backfill_keypoint_edge_distances import _backfill_run_group


def _make_root(tmp_path):
    root = zarr.open_group(store=tmp_path / "edge_backfill.zarr", mode="w")

    crop = root.create_group("crop_runs").create_group("crop_001")
    crop.create_array("roi_images", shape=(4, 10, 20), chunks=(2, 10, 20), dtype="u1", fill_value=0)

    kp_parent = root.create_group("keypoints_runs")
    kp_parent.attrs["latest"] = "keypoints_001"
    kp_run = kp_parent.create_group("keypoints_001")
    kp_run.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    kp_run.attrs["pose_schema"] = {
        "name": "traditional_v1",
        "nodes": [
            {"id": 0, "name": "swim_bladder"},
            {"id": 1, "name": "eye_left"},
            {"id": 2, "name": "eye_right"},
        ],
        "edges": [[0, 1], [0, 2], [1, 2]],
    }
    kp_run.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
                [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]],
                [[1.0, 1.0], [np.nan, np.nan], [4.0, 5.0]],
                [[0.0, 0.0], [6.0, 0.0], [0.0, 8.0]],
            ],
            dtype=np.float64,
        ),
    )

    refined = root.create_group("refined_keypoints_runs").create_group("refined_001")
    refined.attrs["source_keypoints_run"] = "keypoints_001"
    refined.attrs["source_crop_run"] = "crop_001"
    refined.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
                [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]],
                [[1.0, 1.0], [np.nan, np.nan], [4.0, 5.0]],
                [[0.0, 0.0], [6.0, 0.0], [0.0, 8.0]],
            ],
            dtype=np.float64,
        ),
        chunks=(2, 3, 2),
    )
    refined.create_array("refined_success", data=np.asarray([True, True, False, True], dtype=bool), chunks=(2,))

    return root


def test_backfill_edge_distances_writes_expected_arrays(tmp_path) -> None:
    root = _make_root(tmp_path)
    refined = root["refined_keypoints_runs"]["refined_001"]

    result = _backfill_run_group(
        root,
        refined,
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    assert result.used_source_fallback is False
    assert "edge_pairs" in refined
    assert "edge_distances" in refined
    assert "edge_distances_norm" in refined
    assert "edge_distance_valid" in refined

    assert np.asarray(refined["edge_pairs"][:]).tolist() == [[0, 1], [0, 2], [1, 2]]
    edge_distances = np.asarray(refined["edge_distances"][:], dtype=np.float64)
    edge_valid = np.asarray(refined["edge_distance_valid"][:], dtype=bool)
    assert edge_distances.shape == (4, 3)
    assert edge_valid.shape == (4, 3)
    np.testing.assert_allclose(edge_distances[0], np.array([3.0, 4.0, 5.0]), atol=1e-5)
    assert edge_valid[2].tolist() == [False, False, False]
    assert refined.attrs["edge_distance_source"] == "pose_schema"
    assert refined.attrs["edge_distance_count"] == 3
    assert refined.attrs["edge_distance_normalization"]["mode"] == "roi_diagonal"


def test_backfill_edge_distances_skips_existing_without_overwrite(tmp_path) -> None:
    root = _make_root(tmp_path)
    refined = root["refined_keypoints_runs"]["refined_001"]
    refined.create_array("edge_pairs", data=np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int16))
    refined.create_array("edge_distances", data=np.zeros((4, 3), dtype=np.float32))
    refined.create_array("edge_distances_norm", data=np.zeros((4, 3), dtype=np.float32))
    refined.create_array("edge_distance_valid", data=np.zeros((4, 3), dtype=np.bool_))

    result = _backfill_run_group(
        root,
        refined,
        overwrite_existing=False,
        apply=True,
    )
    assert result.status == "skipped_existing"


def test_backfill_edge_distances_falls_back_to_refined_attrs_when_source_missing(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "edge_backfill_fallback.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_001")
    refined.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    refined.attrs["pose_schema"] = {
        "edges": [[0, 1], [0, 2], [1, 2]],
    }
    refined.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]],
                [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]],
            ],
            dtype=np.float64,
        ),
    )
    refined.create_array("refined_success", data=np.asarray([True, True], dtype=bool))

    result = _backfill_run_group(
        root,
        refined,
        overwrite_existing=False,
        apply=True,
    )

    assert result.status == "ok"
    assert result.used_source_fallback is True
    assert np.asarray(refined["edge_pairs"][:]).tolist() == [[0, 1], [0, 2], [1, 2]]


def test_backfill_edge_distances_returns_no_edges_without_schema(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "edge_backfill_no_edges.zarr", mode="w")
    refined = root.create_group("refined_keypoints_runs").create_group("refined_001")
    refined.create_array(
        "keypoints_roi",
        shape=(2, 4, 2),
        chunks=(2, 4, 2),
        dtype="f8",
        fill_value=np.nan,
    )
    refined.create_array("refined_success", data=np.asarray([True, True], dtype=bool))

    result = _backfill_run_group(
        root,
        refined,
        overwrite_existing=False,
        apply=True,
    )
    assert result.status == "no_edges"

