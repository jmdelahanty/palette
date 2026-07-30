from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.detection.clipped_native_binding import (
    ClippedDetectionArtifactMember,
    bind_clipped_detection_artifacts,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_file
from fisheye.utils.finalize_recording_canonical_detection_benchmark_adapter import (
    ADAPTER_RECEIPT_NAME,
    finalize_recording_canonical_detection_benchmark_adapter,
)


def _write_array(group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=np.ascontiguousarray(values))


def test_rebuilds_current_canonical_store_without_touching_source(
    tmp_path: Path,
) -> None:
    recording_identity = "recording_full"
    analysis_path = tmp_path / "analysis.zarr"
    model_path = tmp_path / "detect.pt"
    model_path.write_bytes(b"pinned-detection-model")
    frames = np.asarray([0, 2], dtype=np.int32)
    boxes = np.asarray(
        [[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.1, 0.1]],
        dtype=np.float64,
    )
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    classes = np.asarray([0, 1], dtype=np.int32)
    counts = np.asarray([1, 0, 1], dtype=np.int32)
    member = ClippedDetectionArtifactMember(
        work_unit_id="clip_000000:camera_1",
        artifact_run_id="detect_clip_000000",
        clip_id="clip_000000",
        clip_index=0,
        camera_serial="1",
        source_width=640,
        source_height=480,
        artifact_manifest_sha256="a" * 64,
        run_group_tree_sha256="b" * 64,
        parent_frame_indices=np.arange(3, dtype=np.int64),
        frame_indices=frames,
        bbox_norm_coords=boxes,
        scores=scores,
        class_ids=classes,
        artifact_row_id=np.arange(2, dtype=np.uint64),
        frame_counts=counts,
        n_detections=counts.copy(),
    )
    bound = bind_clipped_detection_artifacts(
        (member,),
        recording_identity=recording_identity,
        n_frames=3,
        source_width=640,
        source_height=480,
    )

    root = zarr.open_group(str(analysis_path), mode="w", zarr_format=3)
    source = (
        root.require_group("clips")
        .require_group("clip_000000")
        .require_group("cameras")
        .require_group("1")
        .require_group("detect_runs")
        .create_group("detect_clip_000000")
    )
    source.attrs.update(
        {
            "palette_run_completion_contract": "palette.zarr_run_completion.v1",
            "palette_run_completion_status": "complete",
            "source_video_width": 640,
            "source_video_height": 480,
            "model_path": str(model_path),
            "instance_key_backfill_status": "complete",
            "instance_key_backfill_recording_identity": recording_identity,
        }
    )
    _write_array(source, "frame_indices", frames)
    _write_array(source, "bbox_norm_coords", boxes)
    _write_array(source, "scores", scores)
    _write_array(source, "class_ids", classes)
    _write_array(source, "frame_counts", counts)
    _write_array(source, "n_detections", counts.copy())
    _write_array(
        source,
        "instance_key",
        np.asarray(bound.arrays["instances/instance_key"]),
    )
    source_metadata_before = (
        analysis_path
        / ("clips/clip_000000/cameras/1/detect_runs/detect_clip_000000/zarr.json")
    ).read_bytes()

    frame_index_path = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "camera_serial": ["1", "1", "1"],
                "clip_id": ["clip_000000"] * 3,
                "clip_local_frame_index": [0, 1, 2],
                "parent_frame_index": [0, 1, 2],
            }
        ),
        frame_index_path,
    )
    plan_path = tmp_path / "plan.json"
    plan = {
        "recording_id": recording_identity,
        "analysis_zarr": str(analysis_path),
        "work_unit_count": 1,
        "work_units": [
            {
                "clip_id": "clip_000000",
                "clip_index": 0,
                "camera_serial": "1",
                "frame_count": 3,
                "zarr_paths": {
                    "detect_target_group_path": (
                        "clips/clip_000000/cameras/1/detect_runs/detect_clip_000000"
                    )
                },
            }
        ],
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    benchmark_root = tmp_path / ".palette_benchmarks" / "candidate"
    destination = benchmark_root / "canonical_detection.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = finalize_recording_canonical_detection_benchmark_adapter(
        analysis_zarr=analysis_path,
        detection_plan_path=plan_path,
        recording_frame_index=frame_index_path,
        recording_identity=recording_identity,
        expected_model_sha256=sha256_file(model_path),
        expected_n_frames=3,
        expected_n_instances=2,
        destination=destination,
        benchmark_root=benchmark_root,
        run_id="canonical_candidate",
        scratch_parent=scratch,
        coordinate_catalog=True,
    )

    assert result["status"] == "complete"
    assert result["native_clip_binding_validated"] is True
    assert result["selector_eligible"] is False
    assert result["coordinate_catalog"] is True
    assert result["run_manifest_schema_version"] == 3
    assert (destination / ADAPTER_RECEIPT_NAME).is_file()
    published = zarr.open_group(
        str(destination / "detect_runs" / "canonical_candidate"),
        mode="r",
        use_consolidated=False,
    )
    assert published.attrs["run_manifest"]["schema_version"] == 3
    assert published.attrs["run_manifest"]["payload"]["source_evidence_kind"] == (
        "native_detection"
    )
    assert "coordinate_contract" in published.attrs["run_manifest"]["payload"]
    assert (
        published.attrs["run_manifest"]["payload"]["source_evidence"]["source_kind"]
        == "native_detector"
    )
    assert published.attrs["stage_selector_eligible"] is False
    assert (
        source_metadata_before
        == (
            analysis_path
            / ("clips/clip_000000/cameras/1/detect_runs/detect_clip_000000/zarr.json")
        ).read_bytes()
    )
