import zarr

from fisheye.utils.backfill_keypoint_label_names import _backfill_run_group


def test_backfill_run_group_updates_labels_and_pose_schema(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_label_backfill.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["keypoint_labels"] = ["bladder", "eye_left", "eye_right"]
    run.attrs["keypoint_confidence_labels"] = ["bladder", "eye_left", "eye_right"]
    run.attrs["triangle_angle_order"] = ["angle at bladder", "angle at eye_left", "angle at eye_right"]
    run.attrs["pose_schema"] = {
        "name": "traditional_v1",
        "nodes": [
            {"id": 0, "name": "bladder"},
            {"id": 1, "name": "eye_left"},
            {"id": 2, "name": "eye_right"},
        ],
        "edges": [
            [0, 1],
            [0, 2],
        ],
    }

    result = _backfill_run_group(run, apply=True)

    assert result.status == "ok"
    assert run.attrs["keypoint_labels"] == ["swim_bladder", "eye_left", "eye_right"]
    assert run.attrs["keypoint_confidence_labels"] == ["swim_bladder", "eye_left", "eye_right"]
    assert run.attrs["triangle_angle_order"][0] == "angle at swim_bladder"
    assert run.attrs["pose_schema"]["nodes"][0]["name"] == "swim_bladder"
    assert run.attrs["pose_schema"]["edges"] == [[0, 1], [0, 2], [1, 2]]


def test_backfill_run_group_skips_when_already_canonical(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_label_backfill_skip.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    run.attrs["keypoint_confidence_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    run.attrs["pose_schema"] = {
        "name": "traditional_v1",
        "nodes": [
            {"id": 0, "name": "swim_bladder"},
            {"id": 1, "name": "eye_left"},
            {"id": 2, "name": "eye_right"},
        ],
        "edges": [
            [0, 1],
            [0, 2],
            [1, 2],
        ],
    }

    result = _backfill_run_group(run, apply=True)

    assert result.status == "skipped_existing"


def test_backfill_run_group_canonicalizes_reversed_or_duplicate_edges(tmp_path) -> None:
    root = zarr.open_group(store=tmp_path / "kp_label_backfill_edges.zarr", mode="w")
    run = root.create_group("keypoints_runs").create_group("keypoints_001")
    run.attrs["pose_schema"] = {
        "nodes": [
            {"id": 0, "name": "swim_bladder"},
            {"id": 1, "name": "eye_left"},
            {"id": 2, "name": "eye_right"},
        ],
        "edges": [
            [1, 0],
            [2, 0],
            [2, 1],
            [0, 1],
        ],
    }

    result = _backfill_run_group(run, apply=True)

    assert result.status == "ok"
    assert run.attrs["pose_schema"]["edges"] == [[0, 1], [0, 2], [1, 2]]
