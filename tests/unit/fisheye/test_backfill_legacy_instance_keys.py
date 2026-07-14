from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.utils.backfill_legacy_instance_keys import (
    _atomic_add_array,
    _keys_for_stable_row_ids,
    build_plan,
)


class _Array:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)
        self.shape = self._values.shape

    def __getitem__(self, item: object) -> np.ndarray:
        return self._values[item]


class _Group(dict[str, object]):
    def __init__(self, *args: object, attrs: dict[str, object] | None = None, path: str = "") -> None:
        super().__init__(*args)  # type: ignore[arg-type]
        self.attrs = dict(attrs or {})
        self.path = path


def _run_parent(name: str, group: _Group) -> _Group:
    return _Group(
        {name: group},
        attrs={"latest": name, "latest_complete": name},
    )


def _complete(values: dict[str, object], *, attrs: dict[str, object] | None = None, path: str) -> _Group:
    return _Group(
        {name: value if isinstance(value, _Group) else _Array(value) for name, value in values.items()},
        attrs={"palette_run_completion_status": "complete", **dict(attrs or {})},
        path=path,
    )


def _root() -> _Group:
    recording_id = "recording_RedScare"
    detect_bbox = np.asarray(
        [
            [0.10, 0.20, 0.05, 0.06],
            [0.20, 0.30, 0.05, 0.06],
            [0.30, 0.40, 0.05, 0.06],
        ],
        dtype=np.float64,
    )
    detect = _complete(
        {
            "frame_indices": [0, 1, 2],
            "bbox_norm_coords": detect_bbox,
            "class_ids": [0, 0, 0],
        },
        path="detect_runs/detect_1",
    )
    instances = _Group(
        {
            "frame_indices": _Array([0, 2]),
            "bbox_norm_coords": _Array(detect_bbox[[0, 2]]),
            "class_ids": _Array([0, 0]),
            "source_detect_row_index": _Array([0, 2]),
        },
        path="refined_detect_runs/refined_1/instances",
    )
    source_detections = _Group(
        {
            "frame_indices": _Array([0, 1, 2]),
            "bbox_norm_coords": _Array(detect_bbox),
            "source_detect_row_index": _Array([0, 1, 2]),
        },
        path="refined_detect_runs/refined_1/source_detections",
    )
    refined = _complete(
        {"instances": instances, "source_detections": source_detections},
        path="refined_detect_runs/refined_1",
    )

    crop_bbox = np.asarray(
        [
            [0.11, 0.21, 0.05, 0.06],
            detect_bbox[1],
            [0.31, 0.41, 0.05, 0.06],
        ],
        dtype=np.float64,
    )
    crop = _complete(
        {
            "frame_indices": [0, 1, 2],
            "bbox_norm_coords": crop_bbox,
            "source_detect_row_index": [-1, 1, -1],
        },
        path="crop_runs/crop_1",
    )
    keypoints = _complete(
        {"frame_indices": [0, 1, 2], "source_crop_row_ids": [0, 1, 2]},
        path="keypoints_runs/keypoints_1",
    )
    refined_keypoints = _complete(
        {"frame_indices": [2, 0, 1], "source_crop_row_ids": [2, 0, 1]},
        path="refined_keypoints_runs/refined_keypoints_1",
    )
    arena = _complete(
        {"arena_ids": [0, 0, 0]},
        attrs={"source_rowset_path": "crop_runs/crop_1"},
        path="arena_assignment_runs/arena_1",
    )
    tracking = _complete(
        {
            "frame_indices": [2, 0, 1],
            "source_row_indices": [2, 0, 1],
            "track_ids": [0, 0, 0],
        },
        attrs={
            "source_rowset_path": "crop_runs/crop_1",
            "summary_statistics": {"n_rows": 3, "tracking_identity_mode": "legacy_positional"},
        },
        path="tracking_runs/tracking_1",
    )
    return _Group(
        {
            "detect_runs": _run_parent("detect_1", detect),
            "refined_detect_runs": _run_parent("refined_1", refined),
            "crop_runs": _run_parent("crop_1", crop),
            "keypoints_runs": _run_parent("keypoints_1", keypoints),
            "refined_keypoints_runs": _run_parent("refined_keypoints_1", refined_keypoints),
            "arena_assignment_runs": _run_parent("arena_1", arena),
            "tracking_runs": _run_parent("tracking_1", tracking),
        },
        attrs={"recording_id": recording_id},
    )


def test_build_plan_mints_origins_and_propagates_keys() -> None:
    root = _root()
    plan = build_plan(root, zarr_path=Path("/tmp/recording_RedScare_analysis.zarr"))

    detect_keys = plan.array("detect_runs/detect_1", "instance_key").values
    crop_keys = plan.array("crop_runs/crop_1", "instance_key").values
    keypoint_keys = plan.array("keypoints_runs/keypoints_1", "instance_key").values
    refined_keypoint_keys = plan.array(
        "refined_keypoints_runs/refined_keypoints_1", "instance_key"
    ).values
    tracking_keys = plan.array("tracking_runs/tracking_1", "instance_key").values

    assert crop_keys[1] == detect_keys[1]
    assert len(set(int(value) for value in crop_keys)) == 3
    np.testing.assert_array_equal(keypoint_keys, crop_keys)
    np.testing.assert_array_equal(refined_keypoint_keys, crop_keys[[2, 0, 1]])
    np.testing.assert_array_equal(tracking_keys, crop_keys[[2, 0, 1]])
    assert plan.attrs["tracking_runs/tracking_1"]["tracking_identity_mode"] == "instance_key"
    assert (
        plan.attrs["tracking_runs/tracking_1"]["source_rowset_fingerprint_status"]
        == "complete"
    )
    assert (
        plan.attrs["arena_assignment_runs/arena_1"]["source_rowset_fingerprint_status"]
        == "complete"
    )


def test_build_plan_verifies_matching_existing_keys() -> None:
    root = _root()
    detect = root["detect_runs"]["detect_1"]  # type: ignore[index]
    expected = mint_detection_instance_keys(
        recording_identity="recording_RedScare",
        frame_indices=np.asarray([0, 1, 2]),
        bbox_norm_coords=np.asarray(detect["bbox_norm_coords"][:]),  # type: ignore[index]
        class_ids=np.asarray([0, 0, 0]),
    )
    detect["instance_key"] = _Array(expected)  # type: ignore[index]

    plan = build_plan(root, zarr_path=Path("/tmp/recording_RedScare_analysis.zarr"))
    np.testing.assert_array_equal(
        plan.array("detect_runs/detect_1", "instance_key").values,
        expected,
    )


def test_build_plan_rejects_mismatched_existing_keys() -> None:
    root = _root()
    root["detect_runs"]["detect_1"]["instance_key"] = _Array([1, 2, 3])  # type: ignore[index]

    with pytest.raises(ValueError, match="disagrees with deterministic backfill"):
        build_plan(root, zarr_path=Path("/tmp/recording_RedScare_analysis.zarr"))


def test_build_plan_rejects_bad_crop_row_frame_mapping() -> None:
    root = _root()
    root["keypoints_runs"]["keypoints_1"]["frame_indices"] = _Array([0, 2, 1])  # type: ignore[index]

    with pytest.raises(ValueError, match="keypoints source rows do not map"):
        build_plan(root, zarr_path=Path("/tmp/recording_RedScare_analysis.zarr"))


def test_build_plan_inherits_selected_raw_run_sharding() -> None:
    root = _root()
    root["detect_runs"]["detect_1"].attrs.update(  # type: ignore[index]
        {
            "detect_storage_layout": "indexed_sharding_v1",
            "detect_row_shard_rows": 262_144,
        }
    )
    root["keypoints_runs"]["keypoints_1"].attrs.update(  # type: ignore[index]
        {
            "keypoint_storage_layout": "indexed_sharding_v1",
            "keypoint_roi_shard_rows": 65_536,
        }
    )

    plan = build_plan(root, zarr_path=Path("/tmp/recording_RedScare_analysis.zarr"))

    assert plan.array("detect_runs/detect_1", "instance_key").shard_rows == 262_144
    assert plan.array("keypoints_runs/keypoints_1", "instance_key").shard_rows == 65_536
    assert plan.array("crop_runs/crop_1", "instance_key").shard_rows is None


def test_atomic_add_array_writes_sharded_identity_grid(tmp_path: Path) -> None:
    import zarr

    zarr_path = tmp_path / "identity.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    parent = root.create_group("keypoints_runs")
    parent.create_group("run_1")

    outcome = _atomic_add_array(
        zarr_path=zarr_path,
        group_path="keypoints_runs/run_1",
        name="instance_key",
        values=np.asarray([11, 12, 13, 14, 15], dtype=np.uint64),
        chunk_rows=2,
        shard_rows=8,
    )

    written = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)[
        "keypoints_runs/run_1/instance_key"
    ]
    assert outcome == "written"
    assert written.chunks == (2,)
    assert written.shards == (8,)
    np.testing.assert_array_equal(written[:], [11, 12, 13, 14, 15])


def test_keys_for_stable_row_ids_resolves_nonpositional_ids() -> None:
    resolved = _keys_for_stable_row_ids(
        np.asarray([1001, 1002, 1003], dtype=np.uint64),
        np.asarray([20, 10, 40], dtype=np.int64),
        np.asarray([10, 40, 20], dtype=np.int64),
        label="proxy",
    )

    np.testing.assert_array_equal(resolved, [1002, 1003, 1001])


def test_atomic_add_array_replaces_only_explicitly_authorized_legacy_array(
    tmp_path: Path,
) -> None:
    import zarr

    zarr_path = tmp_path / "identity_repair.zarr"
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    group = root.create_group("detect_runs").create_group("run_1")
    group.create_array("instance_key", data=np.asarray([1, 2, 3], dtype=np.uint64))

    outcome = _atomic_add_array(
        zarr_path=zarr_path,
        group_path="detect_runs/run_1",
        name="instance_key",
        values=np.asarray([11, 12, 13], dtype=np.uint64),
        chunk_rows=2,
        shard_rows=8,
        replace_existing=True,
    )

    repaired = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)[
        "detect_runs/run_1/instance_key"
    ]
    assert outcome == "replaced_verified_legacy"
    np.testing.assert_array_equal(repaired[:], [11, 12, 13])
