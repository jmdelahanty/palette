from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.utils.resolve_clipped_refined_detect_collection import (
    build_collection_frame_map,
)


def test_build_collection_frame_map_uses_latest_collection_and_frame_index(tmp_path: Path) -> None:
    zarr_path = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.require_group("refined_detect_runs").attrs["latest_collection"] = "wf"
    collection = root.require_group("experiment_index/finalized_runs/wf")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"recording_dir": str(tmp_path)}),
        encoding="utf-8",
    )
    collection.attrs["plan_path"] = str(plan_path)
    collection.attrs["selected_runs"] = [
        {
            "work_unit_id": "recording_clip_000000_cam2010093",
            "camera_serial": "2010093",
            "clip_id": "clip_000000",
            "detect_run": "detect_wf_clip_000000_cam2010093",
            "detect_quality_run": "detect_quality_wf_clip_000000_cam2010093",
            "refined_detect_run": "refined_detect_wf_clip_000000_cam2010093",
            "detect_group_path": "clips/clip_000000/cameras/2010093/detect_runs/detect_wf_clip_000000_cam2010093",
            "refined_group_path": "clips/clip_000000/cameras/2010093/refined_detect_runs/refined_detect_wf_clip_000000_cam2010093",
        }
    ]
    frame_index_path = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "camera_serial": "2010093",
                    "clip_id": "clip_000000",
                    "recording_frame_id": 1,
                    "parent_frame_index": 0,
                    "clip_local_frame_index": 0,
                },
                {
                    "camera_serial": "2010093",
                    "clip_id": "clip_000000",
                    "recording_frame_id": 2,
                    "parent_frame_index": 1,
                    "clip_local_frame_index": 1,
                },
            ]
        ),
        frame_index_path,
    )
    (tmp_path / "recording_frame_index_manifest.json").write_text(
        json.dumps({"recording_frame_index_path": str(frame_index_path)}),
        encoding="utf-8",
    )

    summary, table = build_collection_frame_map(zarr_path)

    assert summary["status"] == "ok"
    assert summary["collection_id"] == "wf"
    assert summary["mapped_frame_count"] == 2
    assert summary["unselected_frame_pair_count"] == 0
    rows = table.to_pylist()
    assert rows[0]["recording_frame_id"] == 1
    assert rows[1]["clip_local_frame_index"] == 1
    assert rows[0]["refined_group_path"].endswith(
        "refined_detect_wf_clip_000000_cam2010093"
    )
