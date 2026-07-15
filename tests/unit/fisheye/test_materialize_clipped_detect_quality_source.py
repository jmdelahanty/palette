from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.utils.materialize_clipped_detect_quality_source import (
    SOURCE_SCHEMA,
    materialize_clipped_detect_quality_source,
)
from fisheye.utils.plan_clipped_detect_refine_workflow import PLAN_SCHEMA
from fisheye.utils.repair_clipped_detect_quality_source_geometry import (
    REPAIR_SCHEMA,
    repair_clipped_detect_quality_source_geometry,
)


def _raw_group(
    root: zarr.Group,
    path: str,
    frames: list[int],
    keys: list[int],
    *,
    width: int | None = 4512,
    height: int | None = 4512,
) -> None:
    group = root
    for part in path.split("/"):
        group = group.require_group(part)
    group.create_array("frame_indices", data=np.asarray(frames, dtype=np.int32))
    group.create_array(
        "bbox_norm_coords",
        data=np.asarray([[0.1 + 0.1 * index, 0.2, 0.1, 0.1] for index in range(len(frames))]),
    )
    group.create_array("instance_key", data=np.asarray(keys, dtype=np.uint64))
    if width is not None:
        group.attrs["source_video_width"] = width
    if height is not None:
        group.attrs["source_video_height"] = height


def _fixture(
    tmp_path: Path,
    *,
    geometries: tuple[tuple[int | None, int | None], tuple[int | None, int | None]] = (
        (4512, 4512),
        (4512, 4512),
    ),
) -> tuple[Path, Path, Path]:
    zarr_path = tmp_path / "recording.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    paths = [
        "clips/clip_000000/cameras/1/detect_runs/detect_a",
        "clips/clip_000001/cameras/1/detect_runs/detect_b",
    ]
    _raw_group(root, paths[0], [0, 1], [10, 11], width=geometries[0][0], height=geometries[0][1])
    _raw_group(root, paths[1], [0, 1], [12, 13], width=geometries[1][0], height=geometries[1][1])

    frame_index = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "camera_serial": ["1", "1", "1", "1"],
                "clip_id": ["clip_000000", "clip_000000", "clip_000001", "clip_000001"],
                "clip_local_frame_index": [0, 1, 0, 1],
                "parent_frame_index": [0, 1, 2, 3],
            }
        ),
        frame_index,
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": PLAN_SCHEMA,
                "analysis_zarr": str(zarr_path.resolve()),
                "recording_dir": str(tmp_path),
                "workflow_id": "fixture",
                "work_units": [
                    {
                        "clip_id": f"clip_{index:06d}",
                        "clip_index": index,
                        "camera_serial": "1",
                        "frame_count": 2,
                        "run_names": {"detect": f"detect_{'a' if index == 0 else 'b'}"},
                        "zarr_paths": {"detect_target_group_path": paths[index]},
                    }
                    for index in range(2)
                ],
            }
        ),
        encoding="utf-8",
    )
    return zarr_path, frame_index, plan_path


def test_materializes_recording_ordered_indexed_sharded_source(tmp_path: Path) -> None:
    zarr_path, frame_index, plan_path = _fixture(tmp_path)

    report = materialize_clipped_detect_quality_source(
        zarr_path,
        plan_path=plan_path,
        output_run="source_001",
        recording_frame_index=frame_index,
        shard_rows=4,
        inner_rows=2,
        apply=True,
    )

    assert report["status"] == "complete"
    output = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)[
        "detect_collection_sources/source_001"
    ]
    assert output.attrs["schema_id"] == SOURCE_SCHEMA
    assert output["frame_indices"][:].tolist() == [0, 1, 2, 3]
    assert output["instance_key"][:].tolist() == [10, 11, 12, 13]
    assert output["source_clip_indices"][:].tolist() == [0, 0, 1, 1]
    assert output["source_clip_local_frame_indices"][:].tolist() == [0, 1, 0, 1]
    assert output["instance_key"].shards == (4,)
    assert output.attrs["source_slices"][1]["start"] == 2
    assert output.attrs["schema_id"] == "palette.clipped_detect_quality_source.v2"
    assert output.attrs["source_video_width"] == 4512
    assert output.attrs["source_video_height"] == 4512
    assert output.attrs["source_validation"]["full_frame_geometry_uniform"] is True
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert root.attrs["width"] == 4512
    assert root.attrs["height"] == 4512
    assert root.attrs["source_video_metadata"]["width"] == 4512
    assert root.attrs["source_video_metadata"]["height"] == 4512
    assert root["raw_video"].attrs["source_video_width"] == 4512
    assert root["raw_video"].attrs["source_video_height"] == 4512


def test_materializer_rejects_disagreeing_source_geometry(tmp_path: Path) -> None:
    zarr_path, frame_index, plan_path = _fixture(
        tmp_path,
        geometries=((4512, 4512), (4096, 4512)),
    )

    with pytest.raises(ValueError, match="disagree on full-frame geometry"):
        materialize_clipped_detect_quality_source(
            zarr_path,
            plan_path=plan_path,
            output_run="source_bad_geometry",
            recording_frame_index=frame_index,
            apply=False,
        )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert "detect_collection_sources" not in root
    assert "width" not in root.attrs


def test_materializer_rejects_missing_source_geometry(tmp_path: Path) -> None:
    zarr_path, frame_index, plan_path = _fixture(
        tmp_path,
        geometries=((4512, 4512), (None, None)),
    )

    with pytest.raises(ValueError, match="missing required full-frame geometry attrs"):
        materialize_clipped_detect_quality_source(
            zarr_path,
            plan_path=plan_path,
            output_run="source_missing_geometry",
            recording_frame_index=frame_index,
            apply=False,
        )


def test_materializer_rejects_conflicting_parent_geometry(tmp_path: Path) -> None:
    zarr_path, frame_index, plan_path = _fixture(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root.attrs.update({"width": 640, "height": 640})

    with pytest.raises(ValueError, match="disagrees with validated detection geometry"):
        materialize_clipped_detect_quality_source(
            zarr_path,
            plan_path=plan_path,
            output_run="source_parent_geometry_conflict",
            recording_frame_index=frame_index,
            shard_rows=4,
            inner_rows=2,
            apply=True,
        )

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert root.attrs["width"] == 640
    assert root.attrs["height"] == 640
    failed = root["detect_collection_sources/source_parent_geometry_conflict"]
    assert failed.attrs["palette_run_completion_status"] == "failed"


def test_repairs_historical_source_geometry_without_rewriting_arrays(tmp_path: Path) -> None:
    zarr_path, frame_index, plan_path = _fixture(tmp_path)
    materialize_clipped_detect_quality_source(
        zarr_path,
        plan_path=plan_path,
        output_run="source_historical",
        recording_frame_index=frame_index,
        shard_rows=4,
        inner_rows=2,
        apply=True,
    )
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    source = root["detect_collection_sources/source_historical"]
    before_arrays = {
        name: np.asarray(source[name][:]).copy()
        for name in (
            "frame_indices",
            "bbox_norm_coords",
            "instance_key",
            "source_clip_indices",
            "source_clip_local_frame_indices",
            "source_clip_detect_row_index",
        )
    }
    before_digests = dict(source.attrs["decoded_array_sha256"])
    source.attrs["schema_id"] = "palette.clipped_detect_quality_source.v1"
    for key in (
        "source_video_width",
        "source_video_height",
        "width",
        "height",
        "full_frame_geometry_schema",
        "full_frame_geometry_source",
    ):
        del source.attrs[key]
    source.attrs["source_slices"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"source_video_width", "source_video_height"}
        }
        for row in source.attrs["source_slices"]
    ]
    source_validation = dict(source.attrs["source_validation"])
    for key in (
        "source_video_width",
        "source_video_height",
        "full_frame_geometry_uniform",
    ):
        source_validation.pop(key, None)
    source.attrs["source_validation"] = source_validation
    for key in (
        "width",
        "height",
        "source_video_width",
        "source_video_height",
        "full_frame_geometry_schema",
        "full_frame_geometry_source",
        "source_video_metadata",
    ):
        if key in root.attrs:
            del root.attrs[key]
    raw = root["raw_video"]
    for key in tuple(raw.attrs):
        del raw.attrs[key]

    dry_run = repair_clipped_detect_quality_source_geometry(
        zarr_path,
        plan_path=plan_path,
        source_run="source_historical",
        recording_frame_index=frame_index,
        apply=False,
    )
    assert dry_run["status"] == "planned"

    report = repair_clipped_detect_quality_source_geometry(
        zarr_path,
        plan_path=plan_path,
        source_run="source_historical",
        recording_frame_index=frame_index,
        apply=True,
    )
    assert report["status"] == "complete"
    assert report["repair"]["schema"] == REPAIR_SCHEMA
    assert report["repair"]["array_payload_rewritten"] is False
    repaired = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    repaired_source = repaired["detect_collection_sources/source_historical"]
    assert repaired_source.attrs["schema_id"] == "palette.clipped_detect_quality_source.v1"
    assert repaired_source.attrs["source_video_width"] == 4512
    assert repaired_source.attrs["source_video_height"] == 4512
    assert repaired_source.attrs["decoded_array_sha256"] == before_digests
    assert repaired_source.attrs["full_frame_geometry_repair"]["schema"] == REPAIR_SCHEMA
    assert repaired.attrs["width"] == 4512
    assert repaired["raw_video"].attrs["height"] == 4512
    for name, expected in before_arrays.items():
        np.testing.assert_array_equal(repaired_source[name][:], expected)

    repeated = repair_clipped_detect_quality_source_geometry(
        zarr_path,
        plan_path=plan_path,
        source_run="source_historical",
        recording_frame_index=frame_index,
        apply=True,
    )
    assert repeated == report
