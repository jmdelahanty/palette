from __future__ import annotations

import numpy as np
import zarr

from fisheye.detection import detect_yolo as mod
from fisheye.diagnostics.audit_yolo_detection_sharding import (
    audit_detection_runs,
    replay_detection_run_as_sharded,
)


def _detection_values() -> dict[str, np.ndarray]:
    frame_indices = np.asarray([0, 1, 1, 3, 5, 8, 13, 21, 34, 35], dtype=np.int32)
    bbox_coords = np.arange(40, dtype=np.float64).reshape(10, 4) / 100.0
    scores = np.linspace(0.5, 0.95, 10, dtype=np.float32)
    class_ids = np.arange(10, dtype=np.int32) % 2
    instance_keys = np.arange(100, 110, dtype=np.uint64)
    frame_counts = np.bincount(frame_indices, minlength=40_000).astype(np.int32)
    return {
        "frame_indices": frame_indices,
        "bbox_coords": bbox_coords,
        "scores": scores,
        "class_ids": class_ids,
        "instance_keys": instance_keys,
        "frame_counts": frame_counts,
    }


def test_write_detection_output_arrays_uses_complete_indexed_shards(tmp_path) -> None:
    group = zarr.open_group(tmp_path / "detect_sharded.zarr", mode="w")
    values = _detection_values()

    summary = mod._write_detection_output_arrays(  # noqa: SLF001
        group,
        **values,
        det_chunk=4,
        detect_row_shard_rows=8,
        detect_frame_shard_rows=32_768,
    )

    assert summary is not None
    assert summary["schema_id"] == mod.DETECT_SHARD_WRITE_SCHEMA
    assert summary["write_mode"] == "materialized_complete_shards"
    assert summary["detect_row_shard_rows_effective"] == 8
    assert summary["detect_frame_shard_rows_effective"] == 32_768
    assert summary["exact_match"] is True
    assert summary["source_sha256_by_array"] == summary["destination_sha256_by_array"]

    expected = {
        "frame_indices": values["frame_indices"],
        "bbox_norm_coords": values["bbox_coords"],
        "scores": values["scores"],
        "class_ids": values["class_ids"],
        "instance_key": values["instance_keys"],
        "n_detections": values["frame_counts"],
        "frame_counts": values["frame_counts"],
    }
    for name, source in expected.items():
        np.testing.assert_array_equal(group[name][:], source)

    assert group["bbox_norm_coords"].chunks == (4, 4)
    assert group["bbox_norm_coords"].shards == (8, 4)
    assert group["frame_counts"].chunks == (16_384,)
    assert group["frame_counts"].shards == (32_768,)
    assert group["frame_counts"].attrs["storage_profile_id"] == "geometry_preload_v1"


def test_write_detection_output_arrays_preserves_regular_layout(tmp_path) -> None:
    group = zarr.open_group(tmp_path / "detect_regular.zarr", mode="w")
    values = _detection_values()

    summary = mod._write_detection_output_arrays(  # noqa: SLF001
        group,
        **values,
        det_chunk=4,
        detect_row_shard_rows=None,
        detect_frame_shard_rows=32_768,
    )

    assert summary is None
    assert group["bbox_norm_coords"].chunks == (4, 4)
    assert group["bbox_norm_coords"].shards is None
    np.testing.assert_array_equal(group["bbox_norm_coords"][:], values["bbox_coords"])


def test_audit_detection_runs_reports_exact_parity_and_physical_counts(tmp_path) -> None:
    zarr_path = tmp_path / "detect_ab.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    parent = root.create_group("detect_runs")
    regular = parent.create_group("regular")
    sharded = parent.create_group("sharded")
    values = _detection_values()

    mod._write_detection_output_arrays(  # noqa: SLF001
        regular,
        **values,
        det_chunk=4,
        detect_row_shard_rows=None,
        detect_frame_shard_rows=32_768,
    )
    mod._write_detection_output_arrays(  # noqa: SLF001
        sharded,
        **values,
        det_chunk=4,
        detect_row_shard_rows=8,
        detect_frame_shard_rows=32_768,
    )

    report = audit_detection_runs(zarr_path, regular_run="regular", sharded_run="sharded")

    assert report["all_arrays_exact"] is True
    assert all(item["exact"] for item in report["arrays"].values())
    assert report["regular_physical"]["payload_files"] > report["sharded_physical"]["payload_files"]


def test_replay_detection_run_as_sharded_uses_production_writer(tmp_path) -> None:
    zarr_path = tmp_path / "detect_replay.zarr"
    root = zarr.open_group(zarr_path, mode="w")
    parent = root.create_group("detect_runs")
    regular = parent.create_group("regular")
    values = _detection_values()
    mod._write_detection_output_arrays(  # noqa: SLF001
        regular,
        **values,
        det_chunk=4,
        detect_row_shard_rows=None,
        detect_frame_shard_rows=32_768,
    )

    summary = replay_detection_run_as_sharded(
        zarr_path,
        source_run="regular",
        destination_run="replay",
        detect_row_shard_rows=8,
        detect_frame_shard_rows=32_768,
    )
    report = audit_detection_runs(zarr_path, regular_run="regular", sharded_run="replay")

    assert summary["exact_match"] is True
    assert report["all_arrays_exact"] is True
    assert root["detect_runs/replay"].attrs["benchmark_only"] is True
    assert root["detect_runs/replay/bbox_norm_coords"].shards == (8, 4)
