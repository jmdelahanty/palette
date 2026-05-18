from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from fisheye.shared.roi_pixel_contract import (
    ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
    crop_run_pixel_contract,
    orange_mono_pynvvc_luma_pixel_contract,
)
from fisheye.utils.create_clipped_training_zarr import (
    SOURCE_FRAME_INDEX_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,
)
from fisheye.utils.validate_clipped_training_provenance import (
    validate_clipped_training_provenance,
)


def _write_source_frame_index(zarr_path: Path, parent_frame_indices: list[int]) -> None:
    n = len(parent_frame_indices)
    table = pa.table(
        {
            "sample_index": pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
            "session_id": pa.array(["rec_a"] * n),
            "recording_id": pa.array(["rec_a"] * n),
            "producer": pa.array(["test"] * n),
            "recording_folder": pa.array([str(zarr_path.parent.parent)] * n),
            "source_layout": pa.array(["rolling_clips"] * n),
            "recording_backend_mode": pa.array(["test"] * n),
            "camera_serial": pa.array(["2010093"] * n),
            "recording_frame_id": pa.array([idx + 1 for idx in parent_frame_indices], type=pa.int64()),
            "parent_frame_index": pa.array(parent_frame_indices, type=pa.int64()),
            "clip_index": pa.array([0] * n, type=pa.int32()),
            "clip_id": pa.array(["clip_000000"] * n),
            "clip_local_frame_index": pa.array(parent_frame_indices, type=pa.int64()),
            "timestamp": pa.array([1000 + idx for idx in range(n)], type=pa.int64()),
            "timestamp_sys": pa.array([2000 + idx for idx in range(n)], type=pa.int64()),
            "video_path": pa.array([str(zarr_path.parent / "clip.mp4")] * n),
            "metadata_path": pa.array([str(zarr_path.parent / "meta.csv")] * n),
            "keyframe_path": pa.array([str(zarr_path.parent / "keyframe.json")] * n),
            "clip_manifest_path": pa.array([str(zarr_path.parent / "clip_manifest.json")] * n),
            "clip_directory": pa.array(["clips/clip_000000"] * n),
            "clip_recording_folder": pa.array([str(zarr_path.parent / "clip_000000")] * n),
            "source_recording_frame_index_path": pa.array(
                [str(zarr_path.parent / "recording_frame_index.parquet")] * n
            ),
            "frame_step": pa.array([5000] * n, type=pa.int64()),
            "sample_plan_id": pa.array(["sample_plan"] * n),
        }
    )
    pq.write_table(table, zarr_path / "source_frame_index.parquet")


def _make_clipped_training_zarr(tmp_path: Path) -> Path:
    zarr_path = tmp_path / "rec_a_clipped_training.zarr"
    (tmp_path / "recording_frame_index.parquet").write_bytes(b"placeholder")
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=3)
    root.attrs.update(
        {
            "zarr_purpose": "training",
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "source_layout": "rolling_clips",
            "source_frame_index_path": "source_frame_index.parquet",
            "source_frame_index_schema": SOURCE_FRAME_INDEX_SCHEMA_VERSION,
            "source_recording_frame_index_path": str(tmp_path / "recording_frame_index.parquet"),
        }
    )
    _write_source_frame_index(zarr_path, [0, 5000, 10000])

    raw = root.create_group("raw_video")
    raw.create_array("images_full", data=np.zeros((3, 8, 10), dtype=np.uint8), chunks=(3, 8, 10))
    raw.create_array("original_frame_indices", data=np.array([0, 5000, 10000], dtype=np.int64), chunks=(3,))

    crop_parent = root.create_group("crop_runs")
    crop = crop_parent.create_group("crop_luma")
    crop.attrs.update(
        {
            "generated_by": "fisheye.utils.regenerate_training_crops_pynvvc",
            "crop_storage_mode": "materialized",
            "source_layout": "rolling_clips",
            "roi_image_representation": "uint8_grayscale_roi_v1",
            "roi_pixel_contract": orange_mono_pynvvc_luma_pixel_contract(),
            "summary_statistics": {
                "pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
            },
            "provenance": {
                "stage": "crop",
                "parameters": {
                    "roi_pixel_contract": crop_run_pixel_contract(
                        crop_storage_mode="materialized",
                        video_source_type="zarr",
                        acceleration="auto",
                    )
                },
                "inputs": {},
            },
        }
    )
    crop.create_array("source_frame_indices", data=np.array([0, 5000, 10000], dtype=np.int64), chunks=(3,))
    crop.create_array("source_clip_indices", data=np.array([0, 0, 0], dtype=np.int64), chunks=(3,))
    crop.create_array("source_clip_local_frame_indices", data=np.array([0, 5000, 10000], dtype=np.int64), chunks=(3,))
    crop.create_array("roi_images", data=np.zeros((3, 4, 4), dtype=np.uint8), chunks=(3, 4, 4))

    kp_parent = root.create_group("keypoints_runs")
    kp = kp_parent.create_group("keypoints_luma")
    kp.attrs.update(
        {
            "source_crop_run": "crop_luma",
            "source_roi_image_representation": "uint8_grayscale_roi_v1",
            "source_roi_pixel_contract_name": ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME,
            "source_roi_pixel_contract": orange_mono_pynvvc_luma_pixel_contract(),
            "provenance": {
                "inputs": {
                    "source_roi_pixel_contract": orange_mono_pynvvc_luma_pixel_contract(),
                    "source_roi_image_representation": "uint8_grayscale_roi_v1",
                }
            },
        }
    )
    return zarr_path


def test_validate_clipped_training_provenance_plans_stale_nested_contract_repair(tmp_path: Path) -> None:
    zarr_path = _make_clipped_training_zarr(tmp_path)

    report = validate_clipped_training_provenance(zarr_path)

    assert report["status"] == "needs_repair"
    assert report["error_count"] == 0
    assert report["planned_repairs"] == 1
    repair = report["repairs"][0]
    assert repair["target_path"] == "crop_runs/crop_luma"
    assert repair["attr_path"] == ["provenance", "parameters", "roi_pixel_contract"]
    assert repair["new_value"]["name"] == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME


def test_validate_clipped_training_provenance_apply_repairs_nested_contract(tmp_path: Path) -> None:
    zarr_path = _make_clipped_training_zarr(tmp_path)

    apply_report = validate_clipped_training_provenance(zarr_path, apply=True)
    recheck = validate_clipped_training_provenance(zarr_path)

    assert apply_report["status"] == "ok"
    assert apply_report["applied_repairs"] == 1
    assert recheck["status"] == "ok"
    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    contract = root["crop_runs/crop_luma"].attrs["provenance"]["parameters"]["roi_pixel_contract"]
    assert contract["name"] == ORANGE_MONO_PYNVVC_LUMA_CONTRACT_NAME
    assert root.attrs["clipped_training_provenance_repair"]["applied_repairs"] == 1


def test_validate_clipped_training_provenance_fails_parent_frame_mismatch(tmp_path: Path) -> None:
    zarr_path = _make_clipped_training_zarr(tmp_path)
    root = zarr.open_group(str(zarr_path), mode="a", use_consolidated=False)
    root["raw_video/original_frame_indices"][:] = np.array([0, 5001, 10000], dtype=np.int64)

    report = validate_clipped_training_provenance(zarr_path)

    assert report["status"] == "failed"
    assert report["error_count"] == 1
    assert report["findings"][0]["code"] == "original_frame_indices_parent_mismatch"
