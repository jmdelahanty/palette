from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.utils import migrate_immutable_yolo_sharding as migration


def _archive(path: Path) -> Path:
    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    root.attrs["recording_id"] = "recording_fixture"

    detect_parent = root.create_group("detect_runs")
    detect_parent.attrs.update({"latest": "detect_1", "latest_complete": "detect_1"})
    detect = detect_parent.create_group("detect_1")
    detect.attrs["palette_run_completion_status"] = "complete"
    detect.create_array("frame_indices", data=np.asarray([0, 0, 1, 2, 2]), chunks=(1024,))
    detect.create_array(
        "bbox_norm_coords",
        data=np.arange(20, dtype=np.float32).reshape(5, 4),
        chunks=(1024, 4),
    )
    detect.create_array(
        "scores", data=np.linspace(0, 1, 5, dtype=np.float32), chunks=(1024,)
    )
    detect.create_array("frame_counts", data=np.asarray([2, 1, 2]), chunks=(1024,))
    detect.create_array("n_detections", data=np.asarray([2, 1, 2]), chunks=(1024,))

    keypoint_parent = root.create_group("keypoints_runs")
    keypoint_parent.attrs.update({"latest": "keypoint_1", "latest_complete": "keypoint_1"})
    keypoint = keypoint_parent.create_group("keypoint_1")
    keypoint.attrs["palette_run_completion_status"] = "complete"
    keypoint.create_array(
        "keypoints_roi",
        data=np.arange(20, dtype=np.float32).reshape(5, 2, 2),
        chunks=(1024, 2, 2),
    )
    keypoint.create_array(
        "frame_indices", data=np.asarray([0, 0, 1, 2, 2]), chunks=(1024,)
    )
    keypoint.create_array("frame_counts", data=np.asarray([2, 1, 2]), chunks=(1024,))
    keypoint.create_array("n_keypoints", data=np.asarray([2, 2, 2]), chunks=(1024,))
    keypoint.create_array("n_rois", data=np.asarray([2, 1, 2]), chunks=(1024,))
    return path


def test_plan_and_apply_preserve_values_and_selectors(tmp_path: Path) -> None:
    path = _archive(tmp_path / "fixture_analysis.zarr")
    before = zarr.open_group(str(path), mode="r", use_consolidated=False)
    expected_bbox = np.asarray(before["detect_runs/detect_1/bbox_norm_coords"][:])
    expected_keypoints = np.asarray(before["keypoints_runs/keypoint_1/keypoints_roi"][:])

    plan = migration.build_plan(path, stages=("detect", "keypoints"))
    assert all(item.action == "migrate" for stage in plan.stages for item in stage.arrays)
    report = migration.apply_plan(plan)

    root = zarr.open_group(str(path), mode="r", use_consolidated=False)
    assert report["status"] == "complete"
    assert all(
        result["source_sha256"]
        == result["destination_sha256"]
        == result["published_sha256"]
        for stage in report["stages"]
        for result in stage["array_results"]
        if result["action"] == "migrate"
    )
    assert root["detect_runs"].attrs["latest_complete"] == "detect_1"
    assert root["keypoints_runs"].attrs["latest_complete"] == "keypoint_1"
    detect = root["detect_runs/detect_1"]
    keypoint = root["keypoints_runs/keypoint_1"]
    np.testing.assert_array_equal(detect["bbox_norm_coords"][:], expected_bbox)
    np.testing.assert_array_equal(keypoint["keypoints_roi"][:], expected_keypoints)
    assert detect["bbox_norm_coords"].shards == (262_144, 4)
    assert keypoint["keypoints_roi"].shards == (262_144, 2, 2)
    assert detect.attrs["detect_storage_policy"] == "migrated_indexed_sharding_v1"
    assert keypoint.attrs["keypoint_storage_policy"] == "migrated_indexed_sharding_v1"
    assert detect.attrs["palette_run_completion_status"] == "complete"
    assert keypoint.attrs["palette_run_completion_status"] == "complete"

    repeated = migration.build_plan(path, stages=("detect", "keypoints"))
    assert all(
        item.action == "verify_existing_sharded"
        for stage in repeated.stages
        for item in stage.arrays
    )
    run_attrs_before = {
        "detect": dict(detect.attrs),
        "keypoints": dict(keypoint.attrs),
    }

    repeated_report = migration.apply_plan(repeated)

    repeated_root = zarr.open_group(str(path), mode="r", use_consolidated=False)
    assert repeated_report["metadata_consolidation"] == []
    assert dict(repeated_root["detect_runs/detect_1"].attrs) == run_attrs_before["detect"]
    assert dict(repeated_root["keypoints_runs/keypoint_1"].attrs) == run_attrs_before["keypoints"]


def test_publish_failure_restores_ordinary_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _archive(tmp_path / "rollback_analysis.zarr")
    plan = migration.build_plan(path, stages=("detect",))
    original_replace = migration.os.replace
    published_temps = 0

    def fail_second_staged_publish(source: object, destination: object) -> None:
        nonlocal published_temps
        source_name = Path(source).name  # type: ignore[arg-type]
        if source_name.startswith(migration._TEMP_PREFIX):
            published_temps += 1
            if published_temps == 2:
                raise OSError("injected publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(migration.os, "replace", fail_second_staged_publish)
    with pytest.raises(OSError, match="injected publish failure"):
        migration.apply_plan(plan)

    root = zarr.open_group(str(path), mode="r", use_consolidated=False)
    detect = root["detect_runs/detect_1"]
    assert detect["frame_indices"].shards is None
    assert detect["bbox_norm_coords"].shards is None
    assert root.attrs["immutable_yolo_sharding_migration_status"] == "error"
    run_dir = path / "detect_runs" / "detect_1"
    assert not any(
        child.name.startswith(
            (migration._TEMP_PREFIX, migration._BACKUP_PREFIX, migration._FAILED_PREFIX)
        )
        for child in run_dir.iterdir()
    )


def test_repairs_early_canary_noop_provenance_from_writer_artifact(tmp_path: Path) -> None:
    path = _archive(tmp_path / "noop_repair_analysis.zarr")
    migration.apply_plan(migration.build_plan(path, stages=("keypoints",)))
    root = zarr.open_group(str(path), mode="a", use_consolidated=False)
    run = root["keypoints_runs/keypoint_1"]
    original_summary = dict(run.attrs["keypoint_shard_write"])
    provenance = dict(run.attrs.get("provenance") or {})
    provenance["artifacts"] = {
        "keypoint_storage_layout": "indexed_sharding_v1",
        "keypoint_storage_policy": "default_indexed_sharding_v1",
        "keypoint_shard_write": original_summary,
    }
    run.attrs["provenance"] = provenance
    run.attrs.update(
        {
            "keypoint_storage_policy": "migrated_indexed_sharding_v1",
            "keypoint_shard_write": {
                "schema_id": migration.MIGRATION_ID,
                "exact_match": True,
                "source_sha256_by_array": {},
            },
            "keypoint_storage_migration": {
                "schema_id": migration.MIGRATION_ID,
                "exact_match": True,
                "source_sha256_by_array": {},
            },
            "immutable_yolo_sharding_migration_status": "complete",
        }
    )

    repair_plan = migration.build_plan(path, stages=("keypoints",))
    assert repair_plan.stages[0].migrated_arrays == []
    assert repair_plan.stages[0].repair_noop_provenance is True
    migration.apply_plan(repair_plan)

    repaired = zarr.open_group(str(path), mode="r", use_consolidated=False)[
        "keypoints_runs/keypoint_1"
    ]
    assert repaired.attrs["keypoint_storage_policy"] == "default_indexed_sharding_v1"
    assert repaired.attrs["keypoint_shard_write"] == original_summary
    assert "keypoint_storage_migration" not in repaired.attrs
    assert "immutable_yolo_sharding_migration_status" not in repaired.attrs
