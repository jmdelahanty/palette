from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.shared import flat_roi_cache as flat_cache_mod
from fisheye.shared.clipped_collection_flat_roi_cache import (
    CLIPPED_COLLECTION_ROW_INDEX_SCHEMA,
    build_clipped_collection_flat_roi_cache,
)
from fisheye.shared.flat_roi_cache import open_flat_roi_cache
from tests.unit.fisheye.test_flat_roi_cache import _FakePynvvcReader


def _make_clipped_collection_archive(tmp_path: Path) -> tuple[Path, Path, list[np.ndarray], np.ndarray]:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")

    collection_id = "workflow_001"
    refined_path = "clips/clip_000000/cameras/2010093/refined_detect_runs/refined_workflow_001_clip_000000_cam2010093"
    refined = root.require_group(refined_path)
    instances = refined.require_group("instances")
    frame_indices = np.array([2, 0], dtype=np.int32)
    bbox_norm = np.array(
        [
            [4 / 5, 3 / 4, 1 / 5, 1 / 4],
            [1 / 5, 1 / 4, 1 / 5, 1 / 4],
        ],
        dtype=np.float64,
    )
    bbox_img_xyxy = np.array(
        [
            [3.5, 2.5, 4.5, 3.5],
            [0.5, 0.5, 1.5, 1.5],
        ],
        dtype=np.float64,
    )
    instances.create_array("frame_indices", data=frame_indices, overwrite=True)
    instances.create_array("frame_offsets", data=np.array([0, 1, 1, 2], dtype=np.int64), overwrite=True)
    instances.create_array("bbox_norm_coords", data=bbox_norm, overwrite=True)
    instances.create_array("bbox_img_xyxy", data=bbox_img_xyxy, overwrite=True)
    instances.create_array("refined_row_ids", data=np.array([20, 10], dtype=np.int64), overwrite=True)
    instances.create_array("source_kind_codes", data=np.array([1, 1], dtype=np.int8), overwrite=True)
    instances.create_array("manual_edit_flags", data=np.array([False, True]), overwrite=True)
    instances.create_array("source_detect_row_index", data=np.array([7, 3], dtype=np.int32), overwrite=True)
    instances.create_array("instance_key", data=np.array([2007, 2003], dtype=np.uint64), overwrite=True)
    instances.create_array("frame_counts", data=np.array([1, 0, 1], dtype=np.int32), overwrite=True)

    video_path = tmp_path / "clip_000000.mp4"
    collection = root.require_group(f"experiment_index/finalized_runs/{collection_id}")
    collection.attrs.update(
        {
            "schema_version": "palette.refined_detect_clip_collection.v1",
            "collection_id": collection_id,
            "collection_kind": "refined_detect_clip_collection",
            "selected_run_count": 1,
            "selected_runs": [
                {
                    "work_unit_id": "recording_clip_000000_cam2010093",
                    "clip_id": "clip_000000",
                    "clip_index": 0,
                    "camera_serial": "2010093",
                    "frame_count": 3,
                    "refined_detect_run": "refined_workflow_001_clip_000000_cam2010093",
                    "refined_group_path": refined_path,
                    "source": {
                        "video_path": str(video_path),
                        "metadata_path": str(tmp_path / "clip_000000_meta.csv"),
                        "keyframe_path": str(tmp_path / "clip_000000_keyframe.json"),
                    },
                }
            ],
        }
    )
    refined_parent = root.require_group("refined_detect_runs")
    refined_parent.attrs["latest_collection"] = collection_id

    frame_index_path = tmp_path / "recording_frame_index.parquet"
    frame_index = pa.table(
        {
            "camera_serial": ["2010093", "2010093", "2010093"],
            "clip_id": ["clip_000000", "clip_000000", "clip_000000"],
            "clip_local_frame_index": np.array([0, 1, 2], dtype=np.int64),
            "recording_frame_id": np.array([10, 11, 12], dtype=np.int64),
            "parent_frame_index": np.array([9, 10, 11], dtype=np.int64),
            "timestamp": np.array([1.0, 1.1, 1.2], dtype=np.float64),
            "timestamp_sys": np.array([2.0, 2.1, 2.2], dtype=np.float64),
        }
    )
    pq.write_table(frame_index, frame_index_path)

    frames = [
        np.arange(4 * 5, dtype=np.uint8).reshape(4, 5) + np.uint8(frame_idx * 20)
        for frame_idx in range(3)
    ]
    expected = np.stack(
        [
            frames[2][2:4, 3:5],
            frames[0][0:2, 0:2],
        ],
        axis=0,
    )
    return zarr_path, frame_index_path, frames, expected


def test_build_clipped_collection_flat_roi_cache_writes_pixels_and_row_index(
    monkeypatch,
    tmp_path: Path,
) -> None:
    zarr_path, frame_index_path, frames, expected = _make_clipped_collection_archive(tmp_path)
    monkeypatch.setattr(
        flat_cache_mod,
        "_open_pynvvc_luma_reader",
        lambda _video_path: _FakePynvvcReader(frames),
    )
    progress_events: list[dict] = []

    manifest = build_clipped_collection_flat_roi_cache(
        zarr_path=zarr_path,
        collection_id="workflow_001",
        recording_frame_index=frame_index_path,
        manifest_path=tmp_path / "cache" / "workflow_001.flat_roi_cache.json",
        roi_size=(2, 2),
        progress_callback=progress_events.append,
        progress_every_batches=1,
    )

    assert manifest["source"]["source_kind"] == "finalized_clipped_refined_detect_collection"
    assert manifest["source"]["collection_id"] == "workflow_001"
    assert manifest["row_index"]["schema"] == CLIPPED_COLLECTION_ROW_INDEX_SCHEMA
    assert "bbox_norm_cx" in manifest["row_index"]["columns"]
    assert "source_detect_row_index" in manifest["row_index"]["columns"]
    assert "instance_key" in manifest["row_index"]["columns"]
    assert manifest["array"]["shape"] == [2, 2, 2]
    assert manifest["builder"]["decode_backend_effective"] == "pynvvc_luma"
    assert manifest["builder"]["pixel_contract"]["name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert manifest["builder"]["pixel_contract_name"] == "orange_mono_pynvvc_luma_uint8_v1"
    assert progress_events[0]["event"] == "start"
    assert progress_events[-1]["event"] == "complete"

    cache = open_flat_roi_cache(manifest["manifest_path"], expected_archive_path=zarr_path, expected_shape=expected.shape)
    try:
        np.testing.assert_array_equal(cache[:], expected)
    finally:
        cache.close()

    row_index_path = Path(manifest["manifest_path"]).parent / manifest["row_index"]["path"]
    rows = pq.read_table(row_index_path).to_pylist()
    assert [row["roi_row_index"] for row in rows] == [0, 1]
    assert [row["clip_local_frame_index"] for row in rows] == [2, 0]
    assert [row["recording_frame_id"] for row in rows] == [12, 10]
    assert [row["refined_row_id"] for row in rows] == [20, 10]
    assert [row["source_detect_row_index"] for row in rows] == [7, 3]
    assert [row["instance_key"] for row in rows] == [2007, 2003]
    assert rows[0]["roi_x"] == 3
    assert rows[0]["roi_y"] == 2
