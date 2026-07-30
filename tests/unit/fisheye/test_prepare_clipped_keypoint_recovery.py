from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from fisheye.cluster.clipped_inference import LEGACY_PLAN_SCHEMA, PLAN_SCHEMA
from fisheye.utils.prepare_clipped_keypoint_recovery import (
    prepare_keypoint_recovery,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("plan_schema", (LEGACY_PLAN_SCHEMA, PLAN_SCHEMA))
def test_prepare_keypoint_recovery_repairs_proxy_and_removes_only_incomplete(
    tmp_path: Path,
    plan_schema: str,
) -> None:
    zarr_path = tmp_path / "recording_analysis.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.attrs.update({"width": 4512, "height": 4512})
    row_index = tmp_path / "cache" / "clip_000000.parquet"
    row_index.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "bbox_norm_cx": np.asarray([0.25, 0.30], dtype=np.float32),
                "bbox_norm_cy": np.asarray([0.35, 0.40], dtype=np.float32),
                "bbox_norm_w": np.asarray([0.10, 0.10], dtype=np.float32),
                "bbox_norm_h": np.asarray([0.12, 0.12], dtype=np.float32),
                "refined_detect_run": ["refined_clip_000000"] * 2,
                "refined_group_path": ["clips/clip_000000/refined_detect_runs/refined_clip_000000"]
                * 2,
            }
        ),
        row_index,
    )
    proxy_name = "crop_proxy_clip_000000"
    proxy = root.require_group("crop_runs").create_group(proxy_name)
    proxy.create_array(
        "frame_indices",
        data=np.asarray([10, 11], dtype=np.int64),
        chunks=(2,),
    )
    proxy.attrs.update(
        {
            "source_kind": "finalized_clipped_refined_detect_collection_proxy",
            "source_collection_id": "collection_test",
            "source_clip_id": "clip_000000",
            "source_clip_index": 0,
            "source_roi_cache_row_index_path": str(row_index),
        }
    )

    manifest = tmp_path / "cache" / "clip_000000.flat_roi_cache.json"
    alias = tmp_path / "cache" / "clip_000000.alias.json"
    _write_json(
        manifest,
        {"cache_complete": True, "array": {"shape": [2, 512, 512]}},
    )
    _write_json(alias, {"schema": "alias_fixture"})

    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    model_sha = "a" * 64
    mask_name = "subject_mask_shard_clip_000000"
    mask = root.require_group("subject_mask_shard_runs").create_group(mask_name)
    mask.create_array(
        "mask_probs_roi",
        data=np.ones((2, 4, 2, 2), dtype=np.uint8),
        chunks=(2, 4, 2, 2),
    )
    mask.attrs.update(
        {
            "palette_run_completion_status": "complete",
            "source_crop_run": proxy_name,
            "source_collection_id": "collection_test",
            "source_roi_cache_alias_manifest": str(alias),
            "run_provenance": {
                "input_artifacts": [
                    {
                        "role": "subject_mask_unet_checkpoint",
                        "path": str(model),
                        "sha256": model_sha,
                    }
                ]
            },
        }
    )

    keypoint_name = "keypoint_shard_clip_000000"
    keypoints = root.require_group("keypoint_shard_runs")
    incomplete = keypoints.create_group(keypoint_name)
    incomplete.attrs.update(
        {
            "palette_run_name": keypoint_name,
            "output_parent": "keypoint_shard_runs",
            "palette_run_completion_status": "running",
        }
    )
    keypoints.attrs["latest_pending"] = keypoint_name

    package = tmp_path / "packages" / "clip_000000.tar.gz"
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "schema": plan_schema,
            "models": {
                "subject_masks": {"path": str(model), "sha256": model_sha}
            },
            "targets": [
                {
                    "target_id": "recording",
                    "analysis_zarr": str(zarr_path),
                    "collection_id": "collection_test",
                    "merged_proxy_crop_run": "crop_proxy_collection",
                    "keypoint_run": "keypoints_collection",
                    "refined_keypoint_run": "refined_keypoints_collection",
                    "refined_subject_mask_run": "refined_subject_masks_collection",
                    "clips": [
                        {
                            "clip_id": "clip_000000",
                            "proxy_crop_run": proxy_name,
                            "cache_manifest": str(manifest),
                            "alias_manifest": str(alias),
                            "subject_mask_shard_run": mask_name,
                            "keypoint_shard_run": keypoint_name,
                            "package_path": str(package),
                        }
                    ],
                }
            ],
        },
    )

    dry = prepare_keypoint_recovery(plan_path, apply=False)

    assert dry["status"] == "ok"
    assert dry["targets"][0]["merged_proxy_status"] == (
        "will_create_during_keypoint_finalize"
    )
    assert dry["targets"][0]["clips"][0]["keypoint_action"] == "remove_incomplete"
    assert keypoint_name in keypoints

    applied = prepare_keypoint_recovery(plan_path, apply=True)

    assert applied["removed_incomplete_keypoint_group_count"] == 1
    reopened = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    assert keypoint_name not in reopened["keypoint_shard_runs"]
    assert "latest_pending" not in reopened["keypoint_shard_runs"].attrs
    assert reopened[f"crop_runs/{proxy_name}"].attrs["source_video_width"] == 4512
    assert mask_name in reopened["subject_mask_shard_runs"]
