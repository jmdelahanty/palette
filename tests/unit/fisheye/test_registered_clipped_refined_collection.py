from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.shared import flat_roi_cache as flat_cache_mod
from fisheye.shared.clipped_collection_flat_roi_cache import (
    build_clipped_collection_flat_roi_cache,
)
from fisheye.utils.finalize_registered_clipped_refined_collection import (
    COLLECTION_SCHEMA,
    SLICE_MODE,
    finalize_registered_clipped_refined_collection,
)
from tests.unit.fisheye.test_flat_roi_cache import _FakePynvvcReader


def _fixture(tmp_path: Path):
    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    root.attrs.update({"width": 5, "height": 4})
    refined = root.require_group("refined_detect_runs/canonical_refined")
    refined.attrs.update(
        {
            "status": "complete",
            "source_detect_run": "canonical_raw",
            "source_detect_path": "detect_runs/canonical_raw",
            "registered_detection_gate": {
                "requirement": "required",
                "status": "applied",
                "applied": True,
                "gate_run": "gate_exact",
                "gate_digest": "g" * 64,
                "selection_digest": "s" * 64,
                "row_count": 2,
                "rejected_count": 1,
            },
        }
    )
    instances = refined.require_group("instances")
    instances.create_array(
        "frame_indices", data=np.array([0, 3], dtype=np.int32)
    )
    instances.create_array(
        "instance_key", data=np.array([100, 300], dtype=np.uint64)
    )
    instances.create_array(
        "bbox_norm_coords",
        data=np.array(
            [[0.2, 0.25, 0.2, 0.25], [0.8, 0.75, 0.2, 0.25]],
            dtype=np.float32,
        ),
    )
    instances.create_array(
        "bbox_img_xyxy",
        data=np.array([[0.5, 0.5, 1.5, 1.5], [3.5, 2.5, 4.5, 3.5]]),
    )
    instances.create_array(
        "source_detect_row_index", data=np.array([0, 3], dtype=np.int64)
    )
    instances.create_array(
        "confidence_scores", data=np.array([0.9, 0.8], dtype=np.float32)
    )
    instances.create_array("class_ids", data=np.array([0, 0], dtype=np.int32))

    clip0 = tmp_path / "clip0.mp4"
    clip1 = tmp_path / "clip1.mp4"
    clip0.write_bytes(b"clip0")
    clip1.write_bytes(b"clip1")
    frame_index = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "clip_index": np.array([0, 0, 1, 1], dtype=np.int32),
                "camera_serial": ["2010093"] * 4,
                "clip_id": ["clip0", "clip0", "clip1", "clip1"],
                "clip_local_frame_index": np.array([0, 1, 0, 1], dtype=np.int64),
                "recording_frame_id": np.array([1, 2, 3, 4], dtype=np.int64),
                "parent_frame_index": np.array([0, 1, 2, 3], dtype=np.int64),
            }
        ),
        frame_index,
    )
    target = {
        "target_id": "target",
        "native_detection_authority": {
            "recording_identity": "recording",
            "n_frames": 4,
        },
        "registered_dish_geometry": {
            "selection_policy_id": "manual_review_only_v1"
        },
        "clips": [
            {
                "clip_index": 0,
                "clip_id": "clip0",
                "work_unit_id": "clip0:2010093",
                "camera_serial": "2010093",
                "video_path": str(clip0),
            },
            {
                "clip_index": 1,
                "clip_id": "clip1",
                "work_unit_id": "clip1:2010093",
                "camera_serial": "2010093",
                "video_path": str(clip1),
            },
        ],
    }
    return archive, frame_index, target, clip0, clip1


def test_registered_collection_is_immutable_canonical_slice_authority(
    tmp_path: Path,
) -> None:
    archive, frame_index, target, _clip0, _clip1 = _fixture(tmp_path)

    result = finalize_registered_clipped_refined_collection(
        analysis_zarr=archive,
        target=target,
        collection_id="registered_collection",
        refined_run="canonical_refined",
        recording_frame_index=frame_index,
        gate_requirement="required",
        gate_run="gate_exact",
    )

    assert result["status"] == "complete"
    root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
    collection = root["experiment_index/finalized_runs/registered_collection"]
    assert collection.attrs["schema"] == COLLECTION_SCHEMA
    assert collection.attrs["status"] == "complete"
    assert collection.attrs["registered_detection_gate"]["applied"] is True
    selected = collection.attrs["selected_runs"]
    assert [row["source_mode"] for row in selected] == [SLICE_MODE, SLICE_MODE]
    assert [(row["canonical_parent_frame_start"], row["canonical_parent_frame_stop"]) for row in selected] == [
        (0, 2),
        (2, 4),
    ]
    assert [row["row_count"] for row in selected] == [1, 1]
    assert all(
        row["refined_group_path"] == "refined_detect_runs/canonical_refined"
        for row in selected
    )


def test_clipped_cache_slices_canonical_frames_without_duplicate_rows(
    tmp_path: Path, monkeypatch
) -> None:
    archive, frame_index, target, clip0, clip1 = _fixture(tmp_path)
    finalize_registered_clipped_refined_collection(
        analysis_zarr=archive,
        target=target,
        collection_id="registered_collection",
        refined_run="canonical_refined",
        recording_frame_index=frame_index,
        gate_requirement="required",
        gate_run="gate_exact",
    )
    frames = {
        str(clip0): [
            np.arange(20, dtype=np.uint8).reshape(4, 5),
            np.arange(20, dtype=np.uint8).reshape(4, 5) + 20,
        ],
        str(clip1): [
            np.arange(20, dtype=np.uint8).reshape(4, 5) + 40,
            np.arange(20, dtype=np.uint8).reshape(4, 5) + 60,
        ],
    }
    monkeypatch.setattr(
        flat_cache_mod,
        "_open_pynvvc_luma_reader",
        lambda path: _FakePynvvcReader(frames[str(path)]),
    )

    manifests = []
    for clip_id in ("clip0", "clip1"):
        manifests.append(
            build_clipped_collection_flat_roi_cache(
                zarr_path=archive,
                collection_id="registered_collection",
                recording_frame_index=frame_index,
                clip_ids=(clip_id,),
                manifest_path=tmp_path / f"{clip_id}.json",
                roi_size=(2, 2),
            )
        )
    rows = []
    for manifest in manifests:
        row_path = Path(manifest["manifest_path"]).parent / manifest["row_index"]["path"]
        rows.extend(pq.read_table(row_path).to_pylist())

    assert [row["instance_key"] for row in rows] == [100, 300]
    assert [row["clip_local_frame_index"] for row in rows] == [0, 1]
    assert [row["parent_frame_index"] for row in rows] == [0, 3]
    assert [row["refined_instance_row_index"] for row in rows] == [0, 1]
