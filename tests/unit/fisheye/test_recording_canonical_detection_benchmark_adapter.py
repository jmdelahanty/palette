from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from fisheye.shared.zarr.canonical_detection_benchmark_input import (
    build_canonical_detection_benchmark_input,
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
    canonical_input = build_canonical_detection_benchmark_input(
        {
            "frame_indices": frames,
            "bbox_norm_coords": boxes,
            "scores": scores,
            "class_ids": classes,
        },
        recording_identity=recording_identity,
        frame_count=3,
        source_width=640,
        source_height=480,
    )

    root = zarr.open_group(str(analysis_path), mode="w", zarr_format=3)
    source = root.require_group("detect_runs").create_group("detect_recording")
    source.attrs.update(
        {
            "source_video_width": 640,
            "source_video_height": 480,
            "model_path": str(model_path),
            "summary_statistics": {
                "total_detections": 2,
                "frames_with_zero_detections": 1,
            },
        }
    )
    _write_array(source, "frame_indices", frames)
    _write_array(source, "bbox_norm_coords", boxes)
    _write_array(source, "scores", scores)
    _write_array(source, "class_ids", classes)
    _write_array(source, "frame_counts", counts)
    _write_array(source, "n_detections", counts.copy())
    source_path = analysis_path / "detect_runs" / "detect_recording"
    source_metadata_before = (source_path / "zarr.json").read_bytes()

    anchor = tmp_path / "anchor.zarr"
    anchor_root = zarr.open_group(str(anchor), mode="w", zarr_format=3)
    anchor_group = anchor_root.require_group("detect_runs").create_group("anchor")
    anchor_group.attrs.update(
        {
            "logical_schema": {
                "dimensions": canonical_input.dimensions.as_manifest(),
            },
            "selector_eligible": False,
        }
    )
    anchor_run = anchor_group.create_group("instances")
    for path, values in canonical_input.arrays.items():
        _write_array(anchor_run, path.split("/", 1)[1], np.asarray(values))

    benchmark_root = tmp_path / ".palette_benchmarks" / "candidate"
    destination = benchmark_root / "canonical_detection.zarr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = finalize_recording_canonical_detection_benchmark_adapter(
        source_detection_group_path=source_path,
        recording_identity=recording_identity,
        source_model_artifact=model_path,
        canonical_anchor_archive=anchor,
        canonical_anchor_run_id="anchor",
        expected_model_sha256=sha256_file(model_path),
        expected_n_frames=3,
        destination=destination,
        benchmark_root=benchmark_root,
        run_id="canonical_candidate",
        scratch_parent=scratch,
    )

    assert result["status"] == "complete"
    assert result["canonical_anchor_equal"] is True
    assert result["selector_eligible"] is False
    assert (destination / ADAPTER_RECEIPT_NAME).is_file()
    published = zarr.open_group(
        str(destination / "detect_runs" / "canonical_candidate"),
        mode="r",
        use_consolidated=False,
    )
    assert published.attrs["run_manifest"]["schema_version"] == 3
    assert result["coordinate_catalog"] is True
    assert published.attrs["stage_selector_eligible"] is False
    assert source_metadata_before == (source_path / "zarr.json").read_bytes()
