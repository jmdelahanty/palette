from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.detection import clipped_native_artifact_io as artifact_io


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_loader_revalidates_quarantined_artifact_and_frame_mapping(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    group_path = (
        "clips/clip_000000/cameras/2010093/"
        "detection_artifact_runs/detect_artifact_0"
    )
    run = root.create_group(group_path)
    run.attrs.update(
        {
            "coordinate_contract": "unbound_detection_artifact_v1",
            "artifact_row_id_contract": "palette.detection_artifact_row_id.v1",
            "stage_selector_eligible": False,
            "source_video_width": 640,
            "source_video_height": 480,
            "run_provenance": {
                "input_artifacts": [
                    {"role": "detect_model", "sha256": "3" * 64}
                ]
            },
        }
    )
    run.create_array("frame_indices", data=np.asarray([0, 2], dtype=np.int32))
    run.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [[0.25, 0.5, 0.1, 0.1], [0.75, 0.5, 0.1, 0.1]],
            dtype=np.float64,
        ),
    )
    run.create_array("scores", data=np.asarray([0.9, 0.8], dtype=np.float32))
    run.create_array("class_ids", data=np.asarray([0, 0], dtype=np.int32))
    run.create_array("artifact_row_id", data=np.arange(2, dtype=np.uint64))
    run.create_array("frame_counts", data=np.asarray([1, 0, 1], dtype=np.int32))
    run.create_array("n_detections", data=np.asarray([1, 0, 1], dtype=np.int32))

    frame_index = tmp_path / "recording_frame_index.parquet"
    pq.write_table(
        pa.table(
            {
                "camera_serial": ["2010093"] * 3,
                "clip_id": ["clip_000000"] * 3,
                "clip_local_frame_index": np.arange(3, dtype=np.int64),
                "parent_frame_index": np.arange(10, 13, dtype=np.int64),
            }
        ),
        frame_index,
    )
    receipt = archive / (
        "clips/clip_000000/cameras/2010093/detection_artifact_runs/"
        ".imports/detect_artifact_0_import_receipt.json"
    )
    manifest = {
        "clip_context": {
            "recording_id": "recording:fixture",
            "clip_id": "clip_000000",
            "clip_index": 0,
            "camera_serial": "2010093",
            "clip_camera_key": "clip_000000/camera_2010093",
        },
        "checksums": {"run_group_tree_hash": "b" * 64},
    }
    _write_json(
        receipt,
        {
            "target_archive_path": str(archive.resolve()),
            "target_group_path": group_path,
            "run_name": "detect_artifact_0",
            "manifest_sha256": "a" * 64,
            "manifest": manifest,
        },
    )
    report = tmp_path / "report.json"
    _write_json(
        report,
        {
            "schema": "palette.clipped_detection_work_unit_report.v1",
            "status": "ok",
            "recording_id": "recording:fixture",
            "clip_id": "clip_000000",
            "clip_index": 0,
            "camera_serial": "2010093",
            "target_zarr": str(archive.resolve()),
            "target_group_path": group_path,
            "import": {
                "receipt_path": str(receipt),
                "manifest_sha256": "a" * 64,
            },
            "validation": {"status": "pass"},
        },
    )
    monkeypatch.setattr(
        artifact_io,
        "validate_imported_run_group",
        lambda **_kwargs: {"status": "pass"},
    )

    member, evidence = artifact_io.load_clipped_detection_artifact_member(
        report,
        analysis_zarr=archive,
        recording_frame_index=frame_index,
        recording_identity="recording:fixture",
        source_width=640,
        source_height=480,
    )

    np.testing.assert_array_equal(
        member.parent_frame_indices,
        np.asarray([10, 11, 12], dtype=np.int64),
    )
    assert member.artifact_manifest_sha256 == "a" * 64
    assert member.run_group_tree_sha256 == "b" * 64
    assert evidence["artifact_group_path"] == group_path
    assert evidence["run_provenance"]["input_artifacts"][0]["sha256"] == "3" * 64


def test_loader_binds_whole_video_identity_without_frame_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(archive), mode="w", zarr_format=3)
    group_path = "detection_artifact_runs/detect_artifact_whole"
    run = root.create_group(group_path)
    run.attrs.update(
        {
            "coordinate_contract": "unbound_detection_artifact_v1",
            "artifact_row_id_contract": "palette.detection_artifact_row_id.v1",
            "stage_selector_eligible": False,
            "source_video_width": 640,
            "source_video_height": 480,
            "run_provenance": {
                "input_artifacts": [
                    {"role": "detect_model", "sha256": "3" * 64}
                ]
            },
        }
    )
    run.create_array("frame_indices", data=np.asarray([0, 2], dtype=np.int32))
    run.create_array(
        "bbox_norm_coords",
        data=np.asarray(
            [[0.25, 0.5, 0.1, 0.1], [0.75, 0.5, 0.1, 0.1]],
            dtype=np.float64,
        ),
    )
    run.create_array("scores", data=np.asarray([0.9, 0.8], dtype=np.float32))
    run.create_array("class_ids", data=np.asarray([0, 0], dtype=np.int32))
    run.create_array("artifact_row_id", data=np.arange(2, dtype=np.uint64))
    run.create_array("frame_counts", data=np.asarray([1, 0, 1], dtype=np.int32))
    run.create_array("n_detections", data=np.asarray([1, 0, 1], dtype=np.int32))

    receipt = archive / (
        "detection_artifact_runs/.imports/"
        "detect_artifact_whole_import_receipt.json"
    )
    manifest = {
        "frame_mapping_mode": "identity",
        "clip_context": {
            "recording_id": "recording:fixture",
            "clip_id": "whole_video",
            "clip_index": 0,
            "camera_serial": "2010093",
        },
        "checksums": {"run_group_tree_hash": "b" * 64},
    }
    _write_json(
        receipt,
        {
            "target_archive_path": str(archive.resolve()),
            "target_group_path": group_path,
            "run_name": "detect_artifact_whole",
            "manifest_sha256": "a" * 64,
            "manifest": manifest,
        },
    )
    report = tmp_path / "report.json"
    _write_json(
        report,
        {
            "schema": "palette.clipped_detection_work_unit_report.v1",
            "status": "ok",
            "recording_id": "recording:fixture",
            "clip_id": "whole_video",
            "clip_index": 0,
            "camera_serial": "2010093",
            "recording_frame_index": None,
            "frame_mapping_mode": "identity",
            "target_zarr": str(archive.resolve()),
            "target_group_path": group_path,
            "import": {
                "receipt_path": str(receipt),
                "manifest_sha256": "a" * 64,
            },
            "validation": {"status": "pass"},
        },
    )
    monkeypatch.setattr(
        artifact_io,
        "validate_imported_run_group",
        lambda **_kwargs: {"status": "pass"},
    )

    member, evidence = artifact_io.load_clipped_detection_artifact_member(
        report,
        analysis_zarr=archive,
        recording_frame_index=None,
        recording_identity="recording:fixture",
        n_frames=3,
        source_width=640,
        source_height=480,
    )

    np.testing.assert_array_equal(
        member.parent_frame_indices,
        np.arange(3, dtype=np.int64),
    )
    assert evidence["frame_mapping_mode"] == "identity"
