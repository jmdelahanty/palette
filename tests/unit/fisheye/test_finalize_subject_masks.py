from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.diagnostics.benchmark_subject_mask_full_finalizer import (
    benchmark_subject_mask_full_finalizer,
)
from fisheye.diagnostics.benchmark_subject_mask_collection_worker_init import (
    benchmark_collection_worker_initialization,
)
from fisheye.diagnostics.build_subject_mask_finalizer_ab_fixture import (
    build_finalizer_ab_fixture,
)
from fisheye.refinement import finalize_subject_masks as mod
from fisheye.shared.detect_reason_codec import read_reason_labels, write_reason_columns
from fisheye.shared.mask_store import open_mask_store, write_component_rle_mask_store_from_dense
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_LATEST_COMPLETE_ATTR,
    RUN_STATUS_COMPLETE,
)
from fisheye.shared.zarr.stage_arrays import REFINED_SUBJECT_MASKS_SPEC, validate_run
from fisheye.tune import refined_subject_mask_review as review_mod


def _patch_refined_subject_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        review_mod,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "c" * 40,
            "short_hash": "cccccccc",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        review_mod,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.12"},
            "platform": {
                "hostname": "finalize-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.12.0",
                "machine": "x86_64",
            },
        },
    )


@pytest.mark.parametrize("key", ["shard_runs", "subject_mask_shard_runs", "runs"])
def test_load_shard_names_from_file_accepts_documented_mapping_keys(tmp_path: Path, key: str) -> None:
    path = tmp_path / "shards.json"
    path.write_text(json.dumps({key: ["clip_a", {"run_name": "clip_b"}]}), encoding="utf-8")

    assert mod._load_shard_names_from_file(path) == ["clip_a", "clip_b"]


def _build_probability_root(
    store_path: Path | None = None,
    *,
    keypoint_source_crop_row_ids: np.ndarray | None = None,
    subject_source_crop_row_ids: np.ndarray | None = None,
    write_keypoint_source_crop_row_ids: bool = True,
    write_subject_source_crop_row_ids: bool = True,
) -> zarr.Group:
    root = zarr.open_group(str(store_path), mode="w") if store_path is not None else zarr.group()
    crop_parent = root.create_group("crop_runs")
    crop_parent.attrs["latest"] = "crop_001"
    crop = crop_parent.create_group("crop_001")
    crop.attrs["crop_storage_mode"] = "materialized"
    crop.attrs["crop_signature"] = {"signature_version": 2, "crop_revision": 7}
    crop.attrs["crop_revision"] = 7
    crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_001/review_status"
    crop.create_array("roi_images", data=np.zeros((2, 10, 10), dtype=np.uint8), overwrite=True)
    crop.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)

    kp_parent = root.create_group("refined_keypoints_runs")
    kp_parent.attrs["latest"] = "refined_kp_001"
    kp = kp_parent.create_group("refined_kp_001")
    kp.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    kp.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[5.0, 6.0], [2.0, 2.0], [7.0, 2.0]],
                [[5.0, 6.0], [2.0, 4.0], [7.0, 4.0]],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    kp.create_array("detection_success", data=np.asarray([True, True], dtype=bool), overwrite=True)
    if write_keypoint_source_crop_row_ids:
        kp.create_array(
            "source_crop_row_ids",
            data=np.asarray(
                [0, 1] if keypoint_source_crop_row_ids is None else keypoint_source_crop_row_ids,
                dtype=np.int64,
            ),
            overwrite=True,
        )

    parent = root.create_group("subject_mask_runs")
    parent.attrs["latest"] = "subject_probs_001"
    run = parent.create_group("subject_probs_001")
    run.attrs.update(
        {
            "source_crop_run": "crop_001",
            "source_crop_storage_mode": "materialized",
            "source_crop_signature": {"signature_version": 2, "crop_revision": 7},
            "source_crop_revision": 7,
            "source_detect_review_status_ref": "refined_detect_runs/refined_detect_001/review_status",
            "source_roi_image_representation": "grayscale_uint8",
            "source_roi_pixel_contract_name": "nv12_luma_plane_uint8",
            "source_roi_pixel_contract": {
                "name": "nv12_luma_plane_uint8",
                "decode_backend": "pynvvc_luma",
            },
            "source_roi_read_mode": "flat_bin_roi_cache",
            "roi_cache_policy": "always",
            "source_roi_cache_used": True,
            "source_roi_cache_backend": "flat_bin_v1",
            "source_roi_cache_key": "cache-key-001",
            "source_roi_cache_path": "/scratch/job123/cache.flat_roi_cache.json",
            "source_roi_cache_canonical_path": "/groups/cache/cache.flat_roi_cache.json",
            "source_roi_cache_expected_archive_path": "/groups/recordings/rec/zarr/rec_analysis.zarr",
            "source_roi_live_acceleration_requested": "gpu",
            "source_roi_live_acceleration_effective": "gpu",
            "source_roi_live_gpu_chunk_frames": 64,
            "method": "unet_subject_masks_v1",
            "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
            "label_schema_id": "subject_v1_union_eyes",
            "source_keypoints_run": "refined_kp_001",
            "source_keypoint_run": "refined_kp_001",
            "source_keypoint_group": "refined_keypoints_runs",
            "created_at_utc": "2026-04-01T00:00:00+00:00",
            "probabilities_encoding": "linear_uint8_0_255",
            "mask_probability_threshold": 0.5,
        }
    )
    run.create_array("detection_source", data=np.asarray([0, 0], dtype=np.int8), overwrite=True)
    run.create_array("frame_indices", data=np.asarray([10, 11], dtype=np.int32), overwrite=True)
    run.create_array("detection_indices", data=np.asarray([0, 1], dtype=np.int32), overwrite=True)
    run.create_array("frame_counts", data=np.asarray([1, 1], dtype=np.int32), overwrite=True)
    run.create_array("available_channels", data=np.asarray([True, True, True], dtype=bool), overwrite=True)
    if write_subject_source_crop_row_ids:
        run.create_array(
            "source_crop_row_ids",
            data=np.asarray(
                [0, 1] if subject_source_crop_row_ids is None else subject_source_crop_row_ids,
                dtype=np.int64,
            ),
            overwrite=True,
        )

    probs = np.zeros((2, 3, 10, 10), dtype=np.uint8)
    probs[:, 0, 2:9, 2:9] = 255
    probs[0, 0, 4:6, 4:6] = 0
    probs[0, 0, 0, 0] = 255
    probs[0, 1, 1:4, 1:4] = 255
    probs[0, 1, 1:4, 6:9] = 255
    probs[1, 1, 3:6, 1:4] = 255
    probs[1, 1, 3:6, 6:9] = 255
    probs[:, 2, 5:8, 4:7] = 255
    probs[0, 2, 6, 5] = 0
    run.create_array("mask_probs_roi", data=probs, overwrite=True)
    return root


def _build_sharded_subject_mask_root(zarr_path: Path | None = None) -> zarr.Group:
    root = zarr.open_group(str(zarr_path), mode="w") if zarr_path is not None else zarr.group()
    crop_parent = root.create_group("crop_runs")

    crop_specs = {
        "crop_clip_a": {
            "frame_indices": np.asarray([10], dtype=np.int32),
            "source_frame_indices": np.asarray([10], dtype=np.int32),
            "source_clip_indices": np.asarray([0], dtype=np.int32),
            "source_clip_local_frame_indices": np.asarray([0], dtype=np.int32),
            "source_refined_row_ids": np.asarray([100], dtype=np.int64),
            "source_detect_row_index": np.asarray([1000], dtype=np.int64),
            "detection_indices": np.asarray([0], dtype=np.int32),
            "roi_coordinates_full": np.asarray([[4, 5]], dtype=np.int32),
        },
        "crop_clip_b": {
            "frame_indices": np.asarray([11], dtype=np.int32),
            "source_frame_indices": np.asarray([11], dtype=np.int32),
            "source_clip_indices": np.asarray([1], dtype=np.int32),
            "source_clip_local_frame_indices": np.asarray([0], dtype=np.int32),
            "source_refined_row_ids": np.asarray([101], dtype=np.int64),
            "source_detect_row_index": np.asarray([1001], dtype=np.int64),
            "detection_indices": np.asarray([1], dtype=np.int32),
            "roi_coordinates_full": np.asarray([[14, 15]], dtype=np.int32),
        },
    }
    merged = {
        key: np.concatenate([crop_specs["crop_clip_a"][key], crop_specs["crop_clip_b"][key]], axis=0)
        for key in crop_specs["crop_clip_a"]
    }
    for crop_name, arrays in {**crop_specs, "crop_collection": merged}.items():
        crop = crop_parent.create_group(crop_name)
        crop.attrs["crop_storage_mode"] = "geometry_only"
        crop.attrs["crop_signature"] = {"signature_version": 2, "crop_revision": 9}
        crop.attrs["crop_revision"] = 9
        crop.attrs["detect_review_status_ref"] = "refined_detect_runs/refined_detect_collection/review_status"
        for name, data in arrays.items():
            crop.create_array(name, data=data, overwrite=True)
        crop.create_array("frame_counts", data=np.ones((int(arrays["frame_indices"].shape[0]),), dtype=np.int32), overwrite=True)

    parent = root.create_group("subject_mask_shard_runs")
    parent.attrs["latest"] = "subject_masks_clip_b"
    shard_specs = {
        "subject_masks_clip_a": ("crop_clip_a", 0),
        "subject_masks_clip_b": ("crop_clip_b", 1),
    }
    for shard_name, (crop_name, output_row) in shard_specs.items():
        run = parent.create_group(shard_name)
        run.attrs.update(
            {
                "palette_run_completion_status": "complete",
                "source_crop_run": crop_name,
                "source_crop_storage_mode": "geometry_only",
                "source_crop_signature": {"signature_version": 2, "crop_revision": 9},
                "source_crop_revision": 9,
                "source_detect_review_status_ref": "refined_detect_runs/refined_detect_collection/review_status",
                "method": "unet_subject_masks_v1",
                "mask_labels": ["subject_body", "eyes_union", "swim_bladder"],
                "label_schema_id": "subject_v1_union_eyes",
                "created_at_utc": "2026-04-01T00:00:00+00:00",
                "probabilities_encoding": "linear_uint8_0_255",
                "mask_probability_threshold": 0.5,
            }
        )
        crop = crop_parent[crop_name]
        run.create_array("detection_source", data=np.asarray([0], dtype=np.int8), overwrite=True)
        run.create_array("frame_indices", data=np.asarray(crop["frame_indices"][:], dtype=np.int32), overwrite=True)
        run.create_array("source_frame_indices", data=np.asarray(crop["source_frame_indices"][:], dtype=np.int32), overwrite=True)
        run.create_array("detection_indices", data=np.asarray(crop["detection_indices"][:], dtype=np.int32), overwrite=True)
        run.create_array("frame_counts", data=np.ones((1,), dtype=np.int32), overwrite=True)
        run.create_array("available_channels", data=np.asarray([True, True, True], dtype=bool), overwrite=True)
        run.create_array("source_crop_row_ids", data=np.asarray([0], dtype=np.int64), overwrite=True)
        run.create_array("source_clip_indices", data=np.asarray(crop["source_clip_indices"][:], dtype=np.int32), overwrite=True)
        run.create_array(
            "source_clip_local_frame_indices",
            data=np.asarray(crop["source_clip_local_frame_indices"][:], dtype=np.int32),
            overwrite=True,
        )
        run.create_array("source_refined_row_ids", data=np.asarray(crop["source_refined_row_ids"][:], dtype=np.int64), overwrite=True)
        run.create_array("source_detect_row_index", data=np.asarray(crop["source_detect_row_index"][:], dtype=np.int64), overwrite=True)

        probs = np.zeros((1, 3, 10, 10), dtype=np.uint8)
        if output_row == 0:
            probs[0, 0, 2:6, 2:6] = 255
            probs[0, 2, 5:7, 4:6] = 255
        else:
            probs[0, 0, 4:9, 4:9] = 255
            probs[0, 2, 6:9, 6:8] = 255
        run.create_array("mask_probs_roi", data=probs, overwrite=True)
    return root


def test_finalize_subject_mask_run_from_shard_collection_rebases_to_target_crop(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_sharded_subject_mask_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_shard_runs=["subject_masks_clip_b", "subject_masks_clip_a"],
        target_crop_run="crop_collection",
        refined_run="refined_subject_masks_collection",
        components=["subject_body", "swim_bladder"],
        chunk_size=1,
    )

    assert summary["status"] == "updated"
    assert summary["finalized_from_subject_mask_shards"] is True
    assert summary["source_crop_rebased_from_shards"] is True
    assert summary["source_crop_run"] == "crop_collection"
    assert summary["source_subject_mask_shard_runs"] == ["subject_masks_clip_b", "subject_masks_clip_a"]

    run = root["refined_subject_masks_runs"]["refined_subject_masks_collection"]
    assert run.attrs["collection_finalizer_schema"] == mod.SUBJECT_MASK_SHARD_COLLECTION_FINALIZER_SCHEMA
    assert run.attrs["finalized_from_subject_mask_shards"] is True
    assert run.attrs["source_crop_run"] == "crop_collection"
    assert run.attrs["source_crop_rebase_target_run"] == "crop_collection"
    assert run.attrs["source_component_sources"]["subject_body"]["source_stage"] == "subject_mask_shard_runs"
    np.testing.assert_array_equal(run["source_crop_row_ids"][:], np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(run["frame_indices"][:], np.asarray([10, 11], dtype=np.int32))
    np.testing.assert_array_equal(run["source_clip_indices"][:], np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(run["source_refined_row_ids"][:], np.asarray([100, 101], dtype=np.int64))

    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")
    swim_idx = labels.index("swim_bladder")
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    assert masks.shape == (2, 2, 10, 10)
    assert masks[0, body_idx, 2, 2] == 1
    assert masks[1, body_idx, 8, 8] == 1
    assert masks[0, swim_idx, 5, 4] == 1
    assert masks[1, swim_idx, 8, 7] == 1
    provenance = run["components/subject_body/provenance"].attrs
    assert provenance["source_stage"] == "subject_mask_shard_runs"
    assert provenance["source_probability_path"] == (
        "subject_mask_shard_runs/subject_mask_shard_collection/mask_probs_roi"
    )


def test_finalize_subject_mask_run_from_mixed_shards_requires_target_crop(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_sharded_subject_mask_root()

    with pytest.raises(ValueError, match="requires --target-crop-run"):
        mod.finalize_subject_mask_run(
            root,
            subject_shard_runs=["subject_masks_clip_a", "subject_masks_clip_b"],
            refined_run="refined_subject_masks_collection_missing_target",
            components=["subject_body", "swim_bladder"],
            chunk_size=1,
        )


def test_collection_worker_plan_reuses_parent_rebase_without_identity_map(monkeypatch) -> None:
    root = _build_sharded_subject_mask_root()
    source, collection = mod._load_subject_mask_source(
        root,
        subject_run=None,
        subject_shard_runs=["subject_masks_clip_b", "subject_masks_clip_a"],
        target_crop_run="crop_collection",
    )
    assert collection is not None
    plan = mod._build_collection_worker_plan(collection)
    parent_masks = np.asarray(source.group["mask_probs_roi"][:])

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("worker rebuilt the global crop identity map")

    monkeypatch.setattr(mod, "_resolve_subject_mask_crop_rebase", fail_rebuild)
    worker_source, worker_collection = mod._load_subject_mask_source(
        root,
        subject_run=None,
        subject_shard_runs=["subject_masks_clip_b", "subject_masks_clip_a"],
        target_crop_run="crop_collection",
        collection_worker_plan=plan,
    )

    assert worker_collection is not None
    np.testing.assert_array_equal(worker_source.group["mask_probs_roi"][:], parent_masks)
    np.testing.assert_array_equal(worker_collection.source_crop_row_ids, collection.source_crop_row_ids)
    plan_summary = mod._collection_worker_plan_summary(plan)
    assert plan_summary is not None
    assert plan_summary["global_identity_map_builds"] == 1
    assert plan_summary["worker_identity_map_rebuilds"] == 0
    assert plan_summary["array_bytes"] == 6
    assert set(plan_summary["array_dtypes"].values()) == {"uint8"}


def test_indexed_collection_slice_allocates_only_requested_positions(monkeypatch) -> None:
    values = np.arange(100, dtype=np.int32).reshape(100, 1)
    view = mod._IndexedCollectionArray(
        [values],
        source_indices=np.zeros((100,), dtype=np.uint8),
        local_indices=np.arange(100, dtype=np.uint8),
    )
    original_arange = mod.np.arange

    def bounded_arange(start, stop=None, step=1, **kwargs):
        effective_start = 0 if stop is None else int(start)
        effective_stop = int(start) if stop is None else int(stop)
        assert effective_stop - effective_start <= 5
        return original_arange(start, stop, step, **kwargs)

    monkeypatch.setattr(mod.np, "arange", bounded_arange)
    np.testing.assert_array_equal(view[50:55, :], values[50:55, :])


def test_cli_defaults_to_complete_derived_surfaces() -> None:
    defaults = mod._build_parser().parse_args(["/tmp/analysis.zarr"])
    assert defaults.write_eye_geometry is True
    assert defaults.write_component_contours is True
    assert defaults.write_sampled_component_contours is True

    disabled = mod._build_parser().parse_args(
        [
            "/tmp/analysis.zarr",
            "--no-write-eye-geometry",
            "--no-write-component-contours",
            "--no-write-sampled-component-contours",
        ]
    )
    assert disabled.write_eye_geometry is False
    assert disabled.write_component_contours is False
    assert disabled.write_sampled_component_contours is False


def test_finalize_subject_mask_run_creates_refined_candidates_from_probabilities(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
    )

    assert summary["status"] == "updated"
    assert summary["component_names"] == ["subject_body", "eye_left", "eye_right", "swim_bladder"]
    assert summary["chunk_count"] == 2
    assert summary["chunk_size"] == 1
    assert summary["metric_level"] == "cheap"
    assert summary["write_eye_geometry"] is True
    assert summary["write_component_contours"] is True
    assert summary["write_sampled_component_contours"] is True
    assert summary["retain_source_seeds"] is False
    assert summary["source_seed_masks_status"] == "omitted"
    assert summary["execution_backend"] == "serial_driver"
    assert summary["process_shard_execution_enabled"] is False
    assert summary["worker_process_count"] is None
    assert summary["requested_chunk_size"] == 1
    assert summary["worker_chunk_size"] == 1
    assert summary["chunk_alignment"] == "requested_chunk_size"
    assert summary["common_metric_row_chunk"] == 2
    assert summary["common_metric_component_chunk"] == 4
    assert summary["common_metric_write_policy"] == mod.COMMON_DERIVED_METRIC_WRITE_POLICY
    assert summary["timing_summary"]["chunk_count"] == 2
    assert "finalize_subject_body" in summary["timing_summary"]["phase_seconds"]
    assert "compute_spatial_metrics_subject_body" in summary["timing_summary"]["phase_seconds"]
    assert "write_masks_roi_subject_body" in summary["timing_summary"]["phase_seconds"]
    assert "write_finalization_metrics_subject_body" in summary["timing_summary"]["phase_seconds"]
    assert "compute_hole_metrics_eye_left" in summary["timing_summary"]["phase_seconds"]
    assert "compute_hole_metrics_eye_right" in summary["timing_summary"]["phase_seconds"]
    assert "compute_topology_metrics_eye_left" not in summary["timing_summary"]["phase_seconds"]
    assert "compute_topology_metrics_eye_right" not in summary["timing_summary"]["phase_seconds"]
    assert summary["review_counts"]["subject_body"]["needs_review"] >= 1

    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    parent = root["refined_subject_masks_runs"]

    assert run.attrs[RUN_COMPLETION_CONTRACT_ATTR] == RUN_COMPLETION_CONTRACT
    assert run.attrs[RUN_COMPLETION_STATUS_ATTR] == RUN_STATUS_COMPLETE
    assert parent.attrs[RUN_LATEST_COMPLETE_ATTR] == "refined_subject_masks_smart_001"
    assert parent.attrs["latest"] == "refined_subject_masks_smart_001"
    assert run.attrs["method"] == "smart_finalize_subject_masks_v1"
    assert run.attrs["finalization_semantics"] == "smart_probability_to_refined_candidate"
    assert run.attrs["smart_finalizer_chunk_count"] == 2
    assert run.attrs["smart_finalizer_chunk_size"] == 1
    assert run.attrs["smart_finalizer_finalization_metric_row_chunk"] == 2
    assert (
        run.attrs["smart_finalizer_finalization_metric_write_policy"]
        == mod.FINALIZATION_METRIC_WRITE_POLICY
    )
    assert run.attrs["smart_finalizer_common_metric_row_chunk"] == 2
    assert run.attrs["smart_finalizer_common_metric_component_chunk"] == 4
    assert (
        run.attrs["smart_finalizer_common_metric_write_policy"]
        == mod.COMMON_DERIVED_METRIC_WRITE_POLICY
    )
    assert run.attrs["smart_finalizer_metric_level"] == "cheap"
    assert run.attrs["smart_finalizer_retain_source_seeds"] is False
    assert run.attrs["source_seed_masks_status"] == "omitted"
    assert run.attrs["source_seed_masks_reason"] == "production_default"
    assert run.attrs["smart_finalizer_execution_backend"] == "serial_driver"
    assert run.attrs["process_shard_execution_enabled"] is False
    assert run.attrs["worker_process_count"] is None
    assert run.attrs["requested_chunk_size"] == 1
    assert run.attrs["worker_chunk_size"] == 1
    assert run.attrs["chunk_alignment"] == "requested_chunk_size"
    assert run.attrs["smart_finalizer_timing_summary"]["chunk_count"] == 2
    assert len(run.attrs["smart_finalizer_chunk_timings"]) == 2
    assert run.attrs["source_roi_image_representation"] == "grayscale_uint8"
    assert run.attrs["source_roi_pixel_contract_name"] == "nv12_luma_plane_uint8"
    assert run.attrs["source_roi_pixel_contract"]["decode_backend"] == "pynvvc_luma"
    assert run.attrs["source_roi_read_mode"] == "flat_bin_roi_cache"
    assert run.attrs["roi_cache_policy"] == "always"
    assert run.attrs["source_roi_cache_used"] is True
    assert run.attrs["source_roi_cache_backend"] == "flat_bin_v1"
    assert run.attrs["source_roi_cache_key"] == "cache-key-001"
    assert run.attrs["source_roi_cache_path"] == "/scratch/job123/cache.flat_roi_cache.json"
    assert run.attrs["source_roi_cache_canonical_path"] == "/groups/cache/cache.flat_roi_cache.json"
    assert (
        run.attrs["source_roi_cache_expected_archive_path"]
        == "/groups/recordings/rec/zarr/rec_analysis.zarr"
    )
    assert run.attrs["source_roi_live_acceleration_requested"] == "gpu"
    assert run.attrs["source_roi_live_acceleration_effective"] == "gpu"
    assert run.attrs["source_roi_live_gpu_chunk_frames"] == 64
    np.testing.assert_array_equal(run["source_crop_row_ids"][:], np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        root[f"crop_runs/{run.attrs['source_crop_run']}"]["frame_indices"][run["source_crop_row_ids"][:]],
        run["frame_indices"][:],
    )
    provenance_parameters = run.attrs["provenance"]["parameters"]
    assert provenance_parameters["execution_backend"] == "serial_driver"
    assert provenance_parameters["process_shard_execution_enabled"] is False
    assert provenance_parameters["worker_process_count"] is None
    assert provenance_parameters["requested_chunk_size"] == 1
    assert provenance_parameters["chunk_alignment"] == "requested_chunk_size"
    assert provenance_parameters["chunk_size"] == 1
    assert provenance_parameters["metric_level"] == "cheap"
    assert provenance_parameters["retain_source_seeds"] is False
    assert provenance_parameters["source_seed_masks_status"] == "omitted"
    assert run.attrs["eye_geometry_status"] == "computed"
    assert run.attrs["component_contours_status"] == "computed"
    assert run.attrs["sampled_component_contours_status"] == "computed"
    assert run.attrs["refined_subject_mask_review_status"]["state"] == "pending"
    assert run.attrs["component_review_statuses"]["subject_body"]["state"] == "pending"
    assert run.attrs["summary_statistics"]["rows_total"] == 2

    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")
    eye_left_idx = labels.index("eye_left")
    eye_right_idx = labels.index("eye_right")
    swim_idx = labels.index("swim_bladder")
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    assert masks[0, body_idx, 4, 4] == 1
    assert masks[0, body_idx, 0, 0] == 0
    assert np.count_nonzero(masks[:, eye_left_idx]) > 0
    assert np.count_nonzero(masks[:, eye_right_idx]) > 0
    assert masks[0, swim_idx, 6, 5] == 1

    body_reasons = read_reason_labels(run["components/subject_body"])
    assert body_reasons is not None
    assert "cleanup_thresholded_probability" in str(body_reasons[0])
    assert "cleanup_closed_gaps" in str(body_reasons[0])
    assert "needs_review" in str(body_reasons[0])

    eye_left_reasons = read_reason_labels(run["components/eye_left"])
    assert eye_left_reasons is not None
    assert "assigned_from_eyes_union" in str(eye_left_reasons[0])
    assert "split_by_keypoint" in str(eye_left_reasons[0])

    provenance = run["components/subject_body/provenance"].attrs
    assert provenance["finalization_method"] == "smart_finalize_subject_masks_v1"
    assert provenance["source_binary_derivation"] == "smart_finalize(mask_probs_roi)"
    assert provenance["source_probability_path"] == "subject_mask_runs/subject_probs_001/mask_probs_roi"
    assert provenance["source_probability_threshold"] == pytest.approx(0.5)
    assert "finalization_metrics" in run["components/subject_body"]
    metrics = run["components/subject_body/finalization_metrics"]
    assert metrics.attrs["schema_id"] == "refined_subject_component_finalization_metrics_v1"
    assert metrics.attrs["surface_role"] == "sealed_derived_analysis"
    assert metrics.attrs["write_policy"] == mod.FINALIZATION_METRIC_WRITE_POLICY
    assert metrics.attrs["row_chunk"] == 2
    assert np.asarray(metrics["quality_code"][:], dtype=np.int16).shape == (2,)
    component_metrics = run["components/subject_body/metrics"]
    assert component_metrics.attrs["schema_id"] == "refined_subject_component_mask_metrics_v1"
    assert component_metrics.attrs["qc_schema_id"] == "refined_subject_component_metric_qc_reasons_v1"
    assert component_metrics.attrs["qc_policy"]["component_name"] == "subject_body"
    assert component_metrics.attrs["metric_level"] == "cheap"
    assert run["metrics/area_px"].chunks == (2, 4)
    assert run["metrics/centroid_xy"].chunks == (2, 4, 2)
    assert run["metrics"].attrs["write_policy"] == mod.COMMON_DERIVED_METRIC_WRITE_POLICY
    assert run["components/subject_body/area_px"].chunks == (2,)
    assert run["components/subject_body/source_row_fingerprint"].chunks == (2,)
    assert component_metrics["component_count"].chunks == (2,)
    assert component_metrics.attrs["write_policy"] == mod.COMMON_DERIVED_METRIC_WRITE_POLICY
    assert np.isnan(np.asarray(component_metrics["sigma_noise"][:], dtype=np.float32)[0])
    assert run["components/subject_body"].attrs["source_seed_masks_status"] == "omitted"
    assert "source_seed_masks_roi" not in run["components/subject_body"]
    assert "relations/eye_pair/metrics/separation_px" in run


def test_finalization_metric_shell_uses_large_driver_owned_row_chunks() -> None:
    root = zarr.group()
    component = root.require_group("components").require_group("subject_body")

    mod._create_finalization_metric_shell(  # noqa: SLF001
        component,
        metric_names=("area_px_before",),
        total_rows=20000,
    )

    metrics = component["finalization_metrics"]
    assert metrics["area_px_before"].chunks == (mod.FINALIZATION_METRIC_ROW_CHUNK,)
    assert metrics["quality_code"].chunks == (mod.FINALIZATION_METRIC_ROW_CHUNK,)
    assert metrics["quality_score"].chunks == (mod.FINALIZATION_METRIC_ROW_CHUNK,)
    assert metrics.attrs["row_chunk"] == mod.FINALIZATION_METRIC_ROW_CHUNK
    assert metrics.attrs["write_policy"] == mod.FINALIZATION_METRIC_WRITE_POLICY


def test_common_metric_shells_use_large_driver_owned_chunks_and_keep_live_state_small() -> None:
    root = zarr.group()
    run = root.create_group("run")
    mod._ensure_refined_run_metric_shell(run, total_rows=20000, component_count=4)  # noqa: SLF001
    component = mod._create_component_shell(  # noqa: SLF001
        run,
        component_name="subject_body",
        total_rows=20000,
        height=8,
        width=8,
        retain_source_seeds=False,
    )

    assert run["metrics/area_px"].chunks == (mod.COMMON_DERIVED_METRIC_ROW_CHUNK, 4)
    assert run["metrics/centroid_xy"].chunks == (mod.COMMON_DERIVED_METRIC_ROW_CHUNK, 4, 2)
    assert component["area_px"].chunks == (mod.COMMON_DERIVED_METRIC_ROW_CHUNK,)
    assert component["source_row_fingerprint"].chunks == (mod.COMMON_DERIVED_METRIC_ROW_CHUNK,)
    assert component["metrics/component_count"].chunks == (mod.COMMON_DERIVED_METRIC_ROW_CHUNK,)
    assert component["edit_applied"].chunks == (256,)
    assert component["manual_override"].chunks == (256,)
    assert component["source_row_stale"].chunks == (256,)
    assert component.attrs["derived_metric_write_policy"] == mod.COMMON_DERIVED_METRIC_WRITE_POLICY


def test_finalize_subject_mask_run_rejects_eye_union_keypoint_row_identity_mismatch(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root(keypoint_source_crop_row_ids=np.asarray([1, 0], dtype=np.int64))

    with pytest.raises(ValueError, match="row identity mismatch.*refusing to split eye masks"):
        mod.finalize_subject_mask_run(
            root,
            subject_run="subject_probs_001",
            refined_run="refined_subject_masks_bad_eye_identity_001",
            chunk_size=1,
        )

    refined_parent = root.get("refined_subject_masks_runs")
    assert refined_parent is None or "refined_subject_masks_bad_eye_identity_001" not in refined_parent


def test_finalize_subject_mask_run_subsets_collection_keypoints_by_source_crop_row_ids(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()
    kp = root["refined_keypoints_runs"]["refined_kp_001"]
    del kp["keypoints_roi"]
    del kp["detection_success"]
    del kp["source_crop_row_ids"]
    kp.create_array(
        "source_crop_row_ids",
        data=np.asarray([2, 0, 3, 1], dtype=np.int64),
        overwrite=True,
    )
    kp.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]],
                [[5.0, 6.0], [2.0, 2.0], [7.0, 2.0]],
                [[60.0, 60.0], [60.0, 60.0], [60.0, 60.0]],
                [[5.0, 6.0], [2.0, 4.0], [7.0, 4.0]],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    kp.create_array(
        "detection_success",
        data=np.asarray([False, True, False, True], dtype=bool),
        overwrite=True,
    )

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_subset_keypoints_001",
        chunk_size=1,
    )

    identity = summary["eyes_union_assignment_summary"]["keypoint_mask_row_identity"]
    assert identity["row_identity_check"] == "source_crop_row_ids_subset"
    assert identity["keypoint_rows_available"] == 4
    assert identity["keypoint_rows_selected"] == 2
    assert summary["eyes_union_assignment_summary"]["keypoint_mask_row_identity_check"] == "source_crop_row_ids_subset"

    run = root["refined_subject_masks_runs"]["refined_subject_masks_subset_keypoints_001"]
    assert run.attrs["assignment_keypoint_row_identity_check"] == "source_crop_row_ids_subset"
    labels = list(run.attrs["mask_labels"])
    eye_left_idx = labels.index("eye_left")
    eye_right_idx = labels.index("eye_right")
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    assert np.count_nonzero(masks[:, eye_left_idx]) > 0
    assert np.count_nonzero(masks[:, eye_right_idx]) > 0


def test_finalize_subject_mask_run_rejects_missing_subset_keypoint_source_crop_row(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()
    kp = root["refined_keypoints_runs"]["refined_kp_001"]
    del kp["keypoints_roi"]
    del kp["detection_success"]
    del kp["source_crop_row_ids"]
    kp.create_array(
        "source_crop_row_ids",
        data=np.asarray([0, 2, 3], dtype=np.int64),
        overwrite=True,
    )
    kp.create_array(
        "keypoints_roi",
        data=np.zeros((3, 3, 2), dtype=np.float32),
        overwrite=True,
    )
    kp.create_array(
        "detection_success",
        data=np.asarray([True, True, True], dtype=bool),
        overwrite=True,
    )

    with pytest.raises(ValueError, match="missing 1 source_crop_row_ids.*1"):
        mod.finalize_subject_mask_run(
            root,
            subject_run="subject_probs_001",
            refined_run="refined_subject_masks_missing_subset_keypoint_001",
            chunk_size=1,
        )


def test_finalize_subject_mask_run_requires_probabilities_encoding(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()
    del root["subject_mask_runs"]["subject_probs_001"].attrs["probabilities_encoding"]

    with pytest.raises(ValueError, match="subject_mask_runs/subject_probs_001/mask_probs_roi.*missing.*uint8"):
        mod.finalize_subject_mask_run(
            root,
            subject_run="subject_probs_001",
            refined_run="refined_subject_masks_smart_001",
            chunk_size=1,
        )


def test_finalize_subject_mask_run_can_materialize_component_rle_mask_store(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_rle",
        chunk_size=1,
        mask_storage="dense_and_rle",
    )

    assert summary["status"] == "updated"
    assert summary["mask_storage"] == "dense_and_rle"
    assert summary["mask_rle_summary"]["status"] == "written"
    assert summary["mask_rle_summary"]["layout"] == "component_groups"
    assert summary["mask_rle_summary"]["roundtrip_validation"] == {
        "status": "passed",
        "rows_checked": 2,
        "channels_checked": 4,
        "chunks_checked": 2,
        "row_chunk_size": 1,
    }
    assert "write_component_rle_mask_store" in summary["timing_summary"]["phase_seconds"]

    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_rle"]
    assert run.attrs["mask_storage_encoding"] == "dense_uint8+component_rle_v1"
    assert run.attrs["mask_store_encodings"] == ["dense_uint8", "component_rle_v1"]
    assert run.attrs["mask_rle_materialized"] is True
    assert run.attrs["smart_finalizer_mask_rle_summary"]["roundtrip_validation"]["status"] == "passed"
    assert run["mask_rle"].attrs["schema_id"] == "palette_mask_rle_binary_v1"
    assert run["mask_rle"].attrs["layout"] == "component_groups"

    dense = (np.asarray(run["masks_roi"][:], dtype=np.uint8) > 0).astype(np.uint8)
    store = open_mask_store(run, prefer="rle")
    assert store.encoding == "component_rle_v1"
    np.testing.assert_array_equal(store.read_dense(), dense)
    np.testing.assert_array_equal(
        store.read_dense(rows=[1], channels=["eye_left"]),
        dense[1:2, 1:2],
    )


def test_finalize_subject_mask_run_can_materialize_bitpacked_mask_store(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_bitpacked",
        chunk_size=1,
        mask_storage="dense_and_bitpacked",
    )

    assert summary["status"] == "updated"
    assert summary["mask_storage"] == "dense_and_bitpacked"
    assert summary["mask_bitpacked_summary"]["status"] == "written"
    assert summary["mask_bitpacked_summary"]["layout"] == "packed_width_array"
    assert summary["mask_bitpacked_summary"]["roundtrip_validation"] == {
        "status": "passed",
        "rows_checked": 2,
        "channels_checked": 4,
        "chunks_checked": 2,
        "row_chunk_size": 1,
    }
    assert "write_bitpacked_mask_store" in summary["timing_summary"]["phase_seconds"]

    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_bitpacked"]
    assert run.attrs["mask_storage_encoding"] == "dense_uint8+bitpacked_binary_v1"
    assert run.attrs["mask_store_encodings"] == ["dense_uint8", "bitpacked_binary_v1"]
    assert run.attrs["mask_storage_contract_encoding"] == "dense_uint8_v1+bitpacked_binary_v1"
    assert run.attrs["mask_storage_authority"] == "masks_roi"
    assert run.attrs["editable_mask_surface"] == "masks_roi"
    assert run.attrs["derived_mask_caches_stale"] is False
    assert run.attrs["mask_bitpacked_stale"] is False
    assert run.attrs["mask_bitpacked_materialized"] is True
    assert run.attrs["smart_finalizer_mask_bitpacked_summary"]["roundtrip_validation"]["status"] == "passed"
    assert run.attrs["smart_finalizer_mask_bitpacked_summary"]["storage_chunks"] == [2, 4, 10, 2]
    assert run["mask_bitpacked"].attrs["schema_id"] == "palette_mask_bitpacked_binary_v1"
    assert run["mask_bitpacked"].attrs["layout"] == "packed_width_array"
    assert tuple(int(value) for value in run["mask_bitpacked/masks_packed"].chunks) == (2, 4, 10, 2)

    dense = (np.asarray(run["masks_roi"][:], dtype=np.uint8) > 0).astype(np.uint8)
    store = open_mask_store(run, prefer="bitpacked")
    assert store.encoding == "bitpacked_binary_v1"
    np.testing.assert_array_equal(store.read_dense(), dense)
    np.testing.assert_array_equal(
        store.read_dense(rows=[1], channels=["eye_left"]),
        dense[1:2, 1:2],
    )


def test_finalize_subject_mask_run_reads_compact_source_mask_store(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()
    source = root["subject_mask_runs/subject_probs_001"]
    dense_source = (np.asarray(source["mask_probs_roi"][:], dtype=np.uint8) >= 128).astype(np.uint8)
    dense_array = source.create_array("masks_roi", data=dense_source, overwrite=True)
    write_component_rle_mask_store_from_dense(
        source,
        dense_array,
        component_names=tuple(str(value) for value in source.attrs["mask_labels"]),
        encode_row_chunk_size=1,
    )
    del source["masks_roi"]
    del source["mask_probs_roi"]
    source.attrs["masks_roi_materialized"] = False

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_from_compact_source",
        chunk_size=1,
    )

    assert summary["status"] == "updated"
    run = root["refined_subject_masks_runs/refined_subject_masks_from_compact_source"]
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    assert masks.shape == (2, 4, 10, 10)
    assert np.count_nonzero(masks[:, 0]) > 0
    assert np.count_nonzero(masks[:, 1]) > 0
    assert np.count_nonzero(masks[:, 2]) > 0
    provenance = run["components/subject_body/provenance"].attrs
    assert provenance["source_surface_path"] == "subject_mask_runs/subject_probs_001/mask_rle"
    assert provenance["source_surface_kind"] == "binary"
    assert provenance["source_binary_derivation"] == "smart_finalize(mask_rle)"
    eye_provenance = run["components/eye_left/provenance"].attrs
    assert eye_provenance["source_surface_path"] == "subject_mask_runs/subject_probs_001/mask_rle"
    assert eye_provenance["source_binary_derivation"] == "smart_finalize(mask_rle)"
    validation = validate_run(run, REFINED_SUBJECT_MASKS_SPEC)
    assert validation.valid, validation.errors


def test_finalize_subject_mask_run_rejects_rle_only_mask_store(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    with pytest.raises(ValueError, match="compact-only"):
        mod.finalize_subject_mask_run(
            root,
            subject_run="subject_probs_001",
            refined_run="refined_subject_masks_smart_rle_only",
            chunk_size=1,
            mask_storage="rle_v1",
            write_eye_geometry=True,
        )


def test_finalize_subject_mask_run_rejects_bitpacked_only_mask_store(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    with pytest.raises(ValueError, match="compact-only"):
        mod.finalize_subject_mask_run(
            root,
            subject_run="subject_probs_001",
            refined_run="refined_subject_masks_smart_bitpacked_only",
            chunk_size=1,
            mask_storage="bitpacked_v1",
            write_eye_geometry=True,
        )


def test_refresh_refined_subject_mask_metrics_prefers_dense_when_rle_cache_exists(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_refresh_dense_and_rle",
        chunk_size=1,
        mask_storage="dense_and_rle",
    )

    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_refresh_dense_and_rle"]
    assert "masks_roi" in run
    assert "mask_rle" in run
    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")
    run["metrics/area_px"][:, body_idx] = 0.0
    run["components/subject_body"]["area_px"][:] = 0.0

    summary = mod.refresh_refined_subject_mask_metrics_run(
        root,
        refined_run="refined_subject_masks_smart_refresh_dense_and_rle",
        components=["subject_body"],
        chunk_size=1,
        metric_level="cheap",
    )

    assert summary["components"] == ["subject_body"]
    assert summary["mask_store_encoding"] == "dense_uint8"
    assert summary["mask_storage_surface"] == "masks_roi"
    assert (
        summary["mask_store_path"]
        == "refined_subject_masks_runs/refined_subject_masks_smart_refresh_dense_and_rle/masks_roi"
    )
    assert np.count_nonzero(np.asarray(run["metrics/area_px"][:, body_idx], dtype=np.float32)) > 0
    assert np.count_nonzero(np.asarray(run["components/subject_body"]["area_px"][:], dtype=np.float32)) > 0
    assert summary["timing_summary"]["mask_store_encoding"] == "dense_uint8"
    assert summary["timing_summary"]["mask_storage_surface"] == "masks_roi"
    store = open_mask_store(run, prefer="rle")
    assert store.encoding == "component_rle_v1"


def test_finalize_subject_mask_run_can_retain_source_seed_masks(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_retained_seeds_001",
        chunk_size=1,
        retain_source_seeds=True,
    )

    assert summary["retain_source_seeds"] is True
    assert summary["source_seed_masks_status"] == "retained"
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_retained_seeds_001"]
    assert run.attrs["smart_finalizer_retain_source_seeds"] is True
    assert run.attrs["source_seed_masks_status"] == "retained"
    assert run.attrs["source_seed_masks_reason"] == "retain_source_seeds=true"
    provenance_parameters = run.attrs["provenance"]["parameters"]
    assert provenance_parameters["retain_source_seeds"] is True
    assert provenance_parameters["source_seed_masks_status"] == "retained"

    component = run["components/subject_body"]
    assert component.attrs["source_seed_masks_status"] == "retained"
    assert component.attrs["source_seed_masks_reason"] == "retain_source_seeds=true"
    assert component.attrs["source_seed_masks_schema_id"] == "refined_subject_component_source_seed_masks_v1"
    assert tuple(component["source_seed_masks_roi"].shape) == (2, 10, 10)
    assert np.count_nonzero(np.asarray(component["source_seed_masks_roi"][:], dtype=np.uint8)) > 0


def test_component_spatial_metrics_match_legacy_review_helpers() -> None:
    masks = np.zeros((4, 7, 9), dtype=np.uint8)
    masks[0, 1:4, 2:6] = 1
    masks[1, 0, 0] = 1
    masks[1, 6, 8] = 1
    masks[2, 3:6, 1:3] = 1

    metrics = mod._compute_component_spatial_metrics(masks)
    legacy_present, legacy_area = review_mod._compute_mask_metrics(masks[:, None, :, :])
    legacy_geometry = review_mod._compute_geometry_metrics(masks[:, None, :, :])

    np.testing.assert_array_equal(metrics["mask_present"], legacy_present[:, 0])
    np.testing.assert_allclose(metrics["area_px"], legacy_area[:, 0])
    np.testing.assert_allclose(metrics["centroid_xy"], legacy_geometry["centroid_xy"][:, 0, :])
    np.testing.assert_array_equal(metrics["centroid_valid"], legacy_geometry["centroid_valid"][:, 0])
    np.testing.assert_allclose(metrics["bbox_xyxy"], legacy_geometry["bbox_xyxy"][:, 0, :])
    np.testing.assert_array_equal(metrics["bbox_valid"], legacy_geometry["bbox_valid"][:, 0])


def test_component_hole_metrics_match_legacy_topology_helper() -> None:
    masks = np.zeros((3, 9, 9), dtype=np.uint8)
    masks[0, 1:8, 1:8] = 1
    masks[0, 3:5, 3:5] = 0
    masks[1, 2:6, 2:6] = 1

    hole_metrics = mod._compute_component_hole_metrics(masks)
    legacy = review_mod._compute_component_topology_metrics(masks)

    np.testing.assert_array_equal(hole_metrics["hole_count"], legacy["hole_count"])
    np.testing.assert_allclose(hole_metrics["hole_area_fraction"], legacy["hole_area_fraction"])


def test_assigned_eye_component_metric_seed_matches_assignment_contract() -> None:
    masks = np.zeros((3, 9, 9), dtype=np.uint8)
    masks[0, 1:4, 1:4] = 1
    masks[2, 2:7, 2:7] = 1
    masks[2, 4, 4] = 0

    metrics = mod._component_metrics_from_assigned_eye_masks(masks)

    np.testing.assert_array_equal(metrics["component_count"], np.asarray([1, 0, 1], dtype=np.int32))
    np.testing.assert_allclose(
        metrics["largest_component_fraction"],
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    )


def test_finalize_subject_mask_run_can_write_full_metrics_and_eye_geometry(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_full_001",
        chunk_size=1,
        metric_level="full",
        write_eye_geometry=True,
    )

    assert summary["metric_level"] == "full"
    assert summary["write_eye_geometry"] is True
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_full_001"]
    assert run.attrs["eye_geometry_status"] == "computed"
    assert "relations" in run
    assert run["components/subject_body"].attrs["shape_qc_metrics_status"] == "computed"
    assert run["components/subject_body/metrics"].attrs["metric_level"] == "full"


def test_finalize_subject_mask_run_writes_assignment_reuse_eye_geometry_without_recompute(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    def _raise_if_fallback_recompute(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("eye geometry fallback recompute should not run when assignment geometry is complete")

    monkeypatch.setattr(mod, "write_refined_subject_eye_geometry", _raise_if_fallback_recompute)

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_assignment_reuse_geometry",
        chunk_size=1,
        write_eye_geometry=True,
    )

    assert summary["write_eye_geometry"] is True
    assert summary["eye_geometry_reuse_status"] == "assignment_reuse"
    assert summary["eye_geometry_assignment_geometry_rows"] == summary["roi_count"]
    assert summary["warnings"] == []
    phase_seconds = summary["timing_summary"]["phase_seconds"]
    assert "write_eye_geometry_from_assignment" in phase_seconds
    assert "write_eye_geometry" not in phase_seconds

    run = root["refined_subject_masks_runs"]["refined_subject_masks_assignment_reuse_geometry"]
    assert run.attrs["eye_geometry_status"] == "computed"
    assert run.attrs["eye_geometry_reuse_status"] == "assignment_reuse"
    assert run.attrs["smart_finalizer_warnings"] == []
    assert run.attrs["eye_geometry_postcompute_backend"] == "assignment_reuse"
    assert run.attrs["eye_geometry_source_measurement"] == "eyes_union_assignment_measure_mask"
    assert run["components/eye_left/geometry"].attrs["source_measurement"] == "eyes_union_assignment_measure_mask"
    assert run["components/eye_right/geometry"].attrs["source_measurement"] == "eyes_union_assignment_measure_mask"
    assert run["relations/eye_pair/metrics"].attrs["source_measurement"] == "eyes_union_assignment_measure_mask"


def test_finalize_subject_mask_run_warns_when_eye_geometry_falls_back_to_refit(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    monkeypatch.setattr(mod, "_eye_geometry_from_assignment_result", lambda _assignment: None)

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_fallback_geometry_warning",
        chunk_size=1,
        write_eye_geometry=True,
    )

    assert summary["write_eye_geometry"] is True
    assert summary["eye_geometry_reuse_status"] == "fallback_refit"
    assert summary["eye_geometry_assignment_geometry_rows"] == 0
    assert summary["eye_geometry_assignment_geometry_expected_rows"] == summary["roi_count"]
    assert summary["warnings"][0]["code"] == "EYE_GEOMETRY_FALLBACK_REFIT"
    phase_seconds = summary["timing_summary"]["phase_seconds"]
    assert "write_eye_geometry_from_assignment" not in phase_seconds
    assert "write_eye_geometry" in phase_seconds

    run = root["refined_subject_masks_runs"]["refined_subject_masks_fallback_geometry_warning"]
    assert run.attrs["eye_geometry_status"] == "computed"
    assert run.attrs["eye_geometry_reuse_status"] == "fallback_refit"
    assert run.attrs["smart_finalizer_warnings"][0]["code"] == "EYE_GEOMETRY_FALLBACK_REFIT"


def test_finalize_subject_mask_run_can_write_body_and_swim_contours(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_contours_001",
        chunk_size=1,
        write_component_contours=True,
    )

    assert summary["write_component_contours"] is True
    assert [item["component"] for item in summary["component_contours"]] == ["subject_body", "swim_bladder"]
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_contours_001"]
    assert run.attrs["component_contours_status"] == "computed"
    assert run.attrs["smart_finalizer_write_component_contours"] is True
    assert run.attrs["component_contours_components"] == ["subject_body", "swim_bladder"]
    for component in ("subject_body", "swim_bladder"):
        contours = run[f"components/{component}/contours"]
        assert contours.attrs["schema_id"] == "component_contours_v1"
        assert contours.attrs["source_component"] == component
        assert contours.attrs["source_mask_run"] == "refined_subject_masks_smart_contours_001"
        assert tuple(contours["ptr"].shape) == (2,)
        assert tuple(contours["len"].shape) == (2,)
        assert contours["points_xy"].shape[1] == 2
        assert np.all(np.asarray(contours["len"][:], dtype=np.int32) > 0)


def test_finalize_subject_mask_run_can_write_sampled_contours_without_full_ragged(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_sampled_contours_001",
        chunk_size=1,
        write_eye_geometry=True,
        write_component_contours=False,
        write_sampled_component_contours=True,
        sampled_contour_counts={
            "subject_body": 8,
            "eye_left": 4,
            "eye_right": 4,
            "swim_bladder": 4,
        },
        sampled_contour_row_chunk=2,
    )

    assert summary["write_sampled_component_contours"] is True
    assert summary["write_component_contours"] is False
    assert [item["component"] for item in summary["sampled_component_contours"]] == [
        "subject_body",
        "eye_left",
        "eye_right",
        "swim_bladder",
    ]
    sampled_summaries = {
        item["component"]: item for item in summary["sampled_component_contours"]
    }
    assert sampled_summaries["subject_body"]["sample_count"] == 8
    assert sampled_summaries["subject_body"]["valid_count"] == 2
    assert sampled_summaries["subject_body"]["source_point_count"] > 0
    assert sampled_summaries["subject_body"]["row_chunk"] == 2
    run = root["refined_subject_masks_runs/refined_subject_masks_sampled_contours_001"]
    assert run.attrs["sampled_component_contours_status"] == "computed"
    assert run.attrs["smart_finalizer_sampled_contour_row_chunk"] == 2
    for component, sample_count in {
        "subject_body": 8,
        "eye_left": 4,
        "eye_right": 4,
        "swim_bladder": 4,
    }.items():
        component_group = run[f"components/{component}"]
        assert "contours" not in component_group
        sampled = component_group["sampled_contours"]
        assert sampled.attrs["surface_role"] == "derived_display_cache"
        assert tuple(sampled["points_xy"].shape) == (2, sample_count, 2)
        assert tuple(sampled["points_xy"].chunks) == (2, sample_count, 2)


def test_finalize_subject_masks_can_write_postcompute_with_process_shards(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / "analysis.zarr"
    _build_probability_root(zarr_path)

    serial_summary = mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_postcompute_serial",
        chunk_size=1,
        write_eye_geometry=True,
        write_component_contours=True,
        write_sampled_component_contours=True,
        sampled_contour_counts={"subject_body": 8, "eye_left": 4, "eye_right": 4, "swim_bladder": 4},
        sampled_contour_row_chunk=2,
        defer_registry_status=True,
    )
    postcompute_calls: list[dict[str, object]] = []
    original_postcompute = mod._run_sharded_refined_subject_postcompute

    def _record_postcompute_call(*args, **kwargs):  # noqa: ANN002, ANN003
        postcompute_calls.append(dict(kwargs))
        return original_postcompute(*args, **kwargs)

    monkeypatch.setattr(mod, "_run_sharded_refined_subject_postcompute", _record_postcompute_call)
    sharded_summary = mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_postcompute_sharded",
        chunk_size=1,
        write_eye_geometry=True,
        write_component_contours=True,
        write_sampled_component_contours=True,
        sampled_contour_counts={"subject_body": 8, "eye_left": 4, "eye_right": 4, "swim_bladder": 4},
        sampled_contour_row_chunk=2,
        postcompute_backend="process_shards",
        postcompute_chunk_size=1,
        postcompute_num_workers=1,
        defer_registry_status=True,
    )

    assert serial_summary["postcompute_backend"] == "serial"
    assert sharded_summary["postcompute_backend"] == "process_shards"
    assert sharded_summary["postcompute_summary"]["status"] == "updated"
    assert sharded_summary["postcompute_summary"]["shard_count"] == 2
    assert [item["component"] for item in sharded_summary["component_contours"]] == [
        "subject_body",
        "swim_bladder",
    ]

    root = zarr.open_group(str(zarr_path), mode="r")
    serial = root["refined_subject_masks_runs/refined_subject_masks_postcompute_serial"]
    sharded = root["refined_subject_masks_runs/refined_subject_masks_postcompute_sharded"]

    assert sharded.attrs["smart_finalizer_postcompute_backend"] == "process_shards"
    assert sharded.attrs["smart_finalizer_postcompute_chunk_size"] == 1
    assert sharded.attrs["smart_finalizer_postcompute_num_workers"] == 1
    assert sharded.attrs["eye_geometry_status"] == "computed"
    assert sharded.attrs["eye_geometry_postcompute_backend"] == "assignment_reuse"
    assert sharded.attrs["eye_geometry_source_measurement"] == "eyes_union_assignment_measure_mask"
    assert sharded.attrs["component_contours_status"] == "computed"
    assert sharded.attrs["component_contours_postcompute_backend"] == "process_shards"
    assert "write_eye_geometry_from_assignment" in sharded.attrs["smart_finalizer_timing_summary"]["phase_seconds"]
    assert "postcompute_process_shards" in sharded.attrs["smart_finalizer_timing_summary"]["phase_seconds"]
    assert postcompute_calls
    assert postcompute_calls[0]["write_eye_geometry"] is False
    assert postcompute_calls[0]["write_component_contours"] is True
    assert postcompute_calls[0]["sampled_contour_counts"] == {
        "subject_body": 8,
        "swim_bladder": 4,
    }
    assert "write_sampled_eye_contours_from_assignment" in sharded.attrs[
        "smart_finalizer_timing_summary"
    ]["phase_seconds"]
    assert sharded.attrs["sampled_component_contours_postcompute_backend"] == "mixed"

    for component in ("eye_left", "eye_right"):
        assert sharded[f"components/{component}/geometry"].attrs["source_measurement"] == "eyes_union_assignment_measure_mask"
        np.testing.assert_allclose(
            np.asarray(serial[f"components/{component}/geometry/ellipse_params"][:], dtype=np.float32),
            np.asarray(sharded[f"components/{component}/geometry/ellipse_params"][:], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            np.asarray(serial[f"components/{component}/geometry/ellipse_success"][:], dtype=bool),
            np.asarray(sharded[f"components/{component}/geometry/ellipse_success"][:], dtype=bool),
        )

    np.testing.assert_allclose(
        np.asarray(serial["relations/eye_pair/metrics/separation_px"][:], dtype=np.float32),
        np.asarray(sharded["relations/eye_pair/metrics/separation_px"][:], dtype=np.float32),
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        np.asarray(serial["relations/eye_pair/metrics/separation_valid"][:], dtype=bool),
        np.asarray(sharded["relations/eye_pair/metrics/separation_valid"][:], dtype=bool),
    )

    for component in ("subject_body", "swim_bladder", "eye_left", "eye_right"):
        for array_name in ("ptr", "len", "points_xy"):
            np.testing.assert_array_equal(
                np.asarray(serial[f"components/{component}/contours/{array_name}"][:]),
                np.asarray(sharded[f"components/{component}/contours/{array_name}"][:]),
            )
        for array_name in ("points_xy", "valid", "source_point_count"):
            np.testing.assert_array_equal(
                np.asarray(serial[f"components/{component}/sampled_contours/{array_name}"][:]),
                np.asarray(sharded[f"components/{component}/sampled_contours/{array_name}"][:]),
            )


def test_finalize_subject_mask_run_reuses_raw_component_topology_metrics(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    def fail_topology_recompute(_masks: np.ndarray) -> dict[str, np.ndarray]:
        raise AssertionError("raw component topology metrics should be reused from finalization metrics")

    monkeypatch.setattr(mod, "_compute_component_topology_metrics", fail_topology_recompute)

    summary = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_reuse_metrics_001",
        components=["subject_body"],
        chunk_size=1,
    )

    assert summary["status"] == "updated"
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_reuse_metrics_001"]
    persisted = run["components/subject_body/metrics"]
    finalization = run["components/subject_body/finalization_metrics"]
    np.testing.assert_array_equal(
        np.asarray(persisted["component_count"][:], dtype=np.int32),
        np.asarray(finalization["component_count_after"][:], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(persisted["largest_component_fraction"][:], dtype=np.float32),
        np.asarray(finalization["largest_component_fraction_after"][:], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(persisted["hole_count"][:], dtype=np.int32),
        np.asarray(finalization["hole_count_after"][:], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(persisted["hole_area_fraction"][:], dtype=np.float32),
        np.asarray(finalization["hole_area_fraction_after"][:], dtype=np.float32),
    )


def test_full_finalizer_benchmark_includes_expensive_phases(monkeypatch, tmp_path: Path) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    source_path = tmp_path / "source.zarr"
    _build_probability_root(source_path)

    payload = benchmark_subject_mask_full_finalizer(
        source_path,
        source_run="subject_probs_001",
        start_row=0,
        roi_count=2,
        chunk_size=1,
        write_eye_geometry=True,
        write_component_contours=True,
        mask_storage="dense_and_bitpacked",
        temp_dir=tmp_path / "bench",
        keep_temp=True,
    )

    assert payload["status"] == "ok"
    assert payload["mask_storage"] == "dense_and_bitpacked"
    assert payload["mask_storage_encoding"] == "dense_uint8+bitpacked_binary_v1"
    assert payload["mask_store_encodings"] == ["dense_uint8", "bitpacked_binary_v1"]
    assert payload["masks_roi_materialized"] is True
    assert payload["copy_summary"]["roi_count"] == 2
    assert Path(str(payload["temp_zarr_path"])).exists()
    phase_seconds = payload["phase_seconds"]
    assert "write_eye_geometry_from_assignment" in phase_seconds
    assert "write_component_contours" in phase_seconds
    workflow_phase_seconds = payload["workflow_profile"]["phase_seconds"]
    assert "copy_benchmark_slice" in workflow_phase_seconds
    assert "finalizer_run" in workflow_phase_seconds
    assert payload["summary_statistics"]["rows_total"] == 2
    output_storage = payload["refined_output_storage_stats"]
    assert output_storage["schema"] == "palette_run_group_storage_stats_v1"
    assert output_storage["file_count"] > 0
    assert output_storage["top_level"]["masks_roi"]["file_count"] > 0
    assert any(
        item["path"] == "components/subject_body/finalization_metrics/quality_score"
        for item in output_storage["arrays"]
    )
    root = zarr.open_group(str(payload["temp_zarr_path"]), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_masks_full_finalizer_benchmark"]
    assert "mask_bitpacked/masks_packed" in run
    assert "masks_roi" in run


def test_full_finalizer_benchmark_can_remove_temp_without_profile_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    source_path = tmp_path / "source.zarr"
    _build_probability_root(source_path)

    payload = benchmark_subject_mask_full_finalizer(
        source_path,
        source_run="subject_probs_001",
        start_row=0,
        roi_count=2,
        chunk_size=1,
        write_eye_geometry=True,
        write_component_contours=False,
        temp_dir=tmp_path / "bench",
        keep_temp=False,
    )

    assert payload["status"] == "ok"
    assert payload["temp_removed_after_run"] is True
    assert payload["workflow_profile_jsonl_retained"] is False
    assert payload["workflow_profile"]["profile_jsonl"] is None
    assert not Path(str(payload["temp_work_dir"])).exists()


def test_full_finalizer_benchmark_can_run_sharded_postcompute(monkeypatch, tmp_path: Path) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    source_path = tmp_path / "source.zarr"
    _build_probability_root(source_path)

    payload = benchmark_subject_mask_full_finalizer(
        source_path,
        source_run="subject_probs_001",
        start_row=0,
        roi_count=2,
        chunk_size=1,
        write_eye_geometry=True,
        write_component_contours=True,
        postcompute_mode="sharded",
        postcompute_chunk_size=1,
        postcompute_num_workers=1,
        temp_dir=tmp_path / "bench",
        keep_temp=True,
    )

    assert payload["status"] == "ok"
    assert payload["postcompute_mode"] == "sharded"
    assert payload["finalizer_write_eye_geometry"] is False
    assert payload["finalizer_write_component_contours"] is False
    assert payload["requested_write_eye_geometry"] is True
    assert payload["requested_write_component_contours"] is True
    sharded = payload["sharded_postcompute_summary"]
    assert sharded["status"] == "updated"
    assert sharded["roi_count"] == 2
    assert sharded["shard_count"] == 2
    assert sharded["eye_geometry"]["status"] == "updated"
    assert [item["component"] for item in sharded["component_contours"]] == ["subject_body", "swim_bladder"]
    workflow_phase_seconds = payload["workflow_profile"]["phase_seconds"]
    assert "sharded_postcompute" in workflow_phase_seconds

    root = zarr.open_group(str(payload["temp_zarr_path"]), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_masks_full_finalizer_benchmark"]
    assert run.attrs["eye_geometry_status"] == "computed"
    assert run.attrs["component_contours_status"] == "computed"
    assert "relations/eye_pair/metrics/separation_px" in run
    assert tuple(run["components/subject_body/contours/ptr"].shape) == (2,)
    assert tuple(run["components/eye_left/geometry/ellipse_params"].shape) == (2, 5)


def test_refresh_refined_subject_mask_metrics_updates_metric_qc_reasons(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()
    mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
    )
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")

    edited_body = np.zeros((10, 10), dtype=np.uint8)
    edited_body[1, 1] = 1
    edited_body[8, 8] = 1
    run["masks_roi"][1, body_idx] = edited_body
    write_reason_columns(
        run["components/subject_body"],
        np.asarray(["clean", "manual_correction|needs_review_metric_holes"], dtype=object),
        chunk_size=2,
        include_reason_text=True,
        overwrite=True,
    )

    summary = mod.refresh_refined_subject_mask_metrics_run(
        root,
        refined_run="refined_subject_masks_smart_001",
        components=["subject_body"],
        chunk_size=1,
        metric_level="cheap",
    )

    assert summary["components"] == ["subject_body"]
    assert summary["review_counts"]["subject_body"]["needs_review"] == 1
    assert float(np.asarray(run["metrics/area_px"][1, body_idx], dtype=np.float32)) == pytest.approx(2.0)
    component_metrics = run["components/subject_body/metrics"]
    assert int(np.asarray(component_metrics["component_count"][1], dtype=np.int32)) == 2
    assert component_metrics.attrs["schema_id"] == "refined_subject_component_mask_metrics_v1"
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    assert run.attrs["component_metric_qc_review_counts"]["subject_body"]["needs_review"] == 1

    body_reasons = read_reason_labels(run["components/subject_body"])
    assert body_reasons is not None
    assert "manual_correction" in str(body_reasons[1])
    assert "needs_review_metric_holes" not in str(body_reasons[1])
    assert "needs_review_metric_small_area" in str(body_reasons[1])
    assert "needs_review_metric_multiple_components" in str(body_reasons[1])


def test_finalize_subject_masks_process_shards_writes_disjoint_rows(monkeypatch, tmp_path: Path) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / "analysis.zarr"
    progress_path = tmp_path / "progress.jsonl"
    _build_probability_root(zarr_path)

    summary = mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_process_shards_001",
        chunk_size=1,
        execution_backend="process_shards",
        num_workers=2,
        progress_jsonl=progress_path,
    )
    mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_serial_reference_001",
        chunk_size=1,
        execution_backend="serial_driver",
    )

    assert summary["execution_backend"] == "process_shards"
    assert summary["process_shard_execution_enabled"] is True
    assert summary["timing_summary"]["process_shard_execution_enabled"] is True
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_masks_smart_process_shards_001"]
    serial = root["refined_subject_masks_runs/refined_subject_masks_smart_serial_reference_001"]
    assert run.attrs["smart_finalizer_execution_backend"] == "process_shards"
    assert run.attrs["process_shard_execution_enabled"] is True
    provenance_parameters = run.attrs["provenance"]["parameters"]
    assert provenance_parameters["execution_backend"] == "process_shards"
    assert provenance_parameters["process_shard_execution_enabled"] is True
    assert provenance_parameters["worker_process_count"] == 2
    assert provenance_parameters["requested_chunk_size"] == 1
    assert provenance_parameters["worker_chunk_size"] == 2
    assert provenance_parameters["chunk_alignment"] == "dense_mask_row_chunk"
    assert provenance_parameters["common_metric_row_chunk"] == 2
    assert provenance_parameters["common_metric_component_chunk"] == 4
    assert provenance_parameters["common_metric_write_policy"] == mod.COMMON_DERIVED_METRIC_WRITE_POLICY
    assert provenance_parameters["finalization_metric_row_chunk"] == 2
    assert provenance_parameters["finalization_metric_write_policy"] == mod.FINALIZATION_METRIC_WRITE_POLICY
    assert len(run.attrs["smart_finalizer_chunk_timings"]) == 1
    labels = list(run.attrs["mask_labels"])
    masks = np.asarray(run["masks_roi"][:], dtype=np.uint8)
    assert np.count_nonzero(masks[:, labels.index("subject_body")]) > 0
    assert np.count_nonzero(masks[:, labels.index("eye_left")]) > 0
    assert np.count_nonzero(masks[:, labels.index("eye_right")]) > 0
    assert np.count_nonzero(masks[:, labels.index("swim_bladder")]) > 0
    assert "process_shard_compute" in run.attrs["smart_finalizer_timing_summary"]["phase_seconds"]
    assert "write_common_run_metrics" in run.attrs["smart_finalizer_timing_summary"]["phase_seconds"]
    assert "write_common_metrics_subject_body" in run.attrs["smart_finalizer_timing_summary"]["phase_seconds"]
    assert "write_run_spatial_metrics_subject_body" not in run.attrs["smart_finalizer_timing_summary"]["phase_seconds"]
    assert "write_finalization_metrics_subject_body" in run.attrs["smart_finalizer_timing_summary"]["phase_seconds"]
    assert (
        run["components/subject_body/finalization_metrics"].attrs["write_policy"]
        == mod.FINALIZATION_METRIC_WRITE_POLICY
    )
    for array_name in ("mask_present", "area_px", "centroid_xy", "centroid_valid", "bbox_xyxy", "bbox_valid"):
        np.testing.assert_allclose(
            np.asarray(run[f"metrics/{array_name}"][:]),
            np.asarray(serial[f"metrics/{array_name}"][:]),
            equal_nan=True,
            err_msg=f"metrics/{array_name}",
        )
    for component_name in labels:
        for array_name in ("mask_present", "area_px", "source_row_fingerprint"):
            np.testing.assert_allclose(
                np.asarray(run[f"components/{component_name}/{array_name}"][:]),
                np.asarray(serial[f"components/{component_name}/{array_name}"][:]),
                equal_nan=True,
                err_msg=f"components/{component_name}/{array_name}",
            )
        for metric_name in mod._COMPONENT_METRIC_NAMES:  # noqa: SLF001
            np.testing.assert_allclose(
                np.asarray(run[f"components/{component_name}/metrics/{metric_name}"][:]),
                np.asarray(serial[f"components/{component_name}/metrics/{metric_name}"][:]),
                equal_nan=True,
                err_msg=f"components/{component_name}/metrics/{metric_name}",
            )

    progress_records = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    aggregate_events = [item for item in progress_records if item["event"] == "process_shards_submitted"]
    shard_events = [item for item in progress_records if item["event"] == "process_shard_submitted"]
    shard_chunk_events = [item for item in progress_records if item["event"] == "process_shard_chunk_completed"]
    assert aggregate_events
    aggregate = aggregate_events[-1]
    expected_shards = int(aggregate["shard_count"])
    assert len(shard_events) == expected_shards
    assert {int(item["shard_index"]) for item in shard_events} == set(range(expected_shards))
    assert all(int(item["total_shards"]) == expected_shards for item in shard_events)
    assert all(int(item["worker_count"]) == expected_shards for item in shard_events)
    row_ranges = sorted((int(item["start_row"]), int(item["stop_row"])) for item in shard_events)
    assert row_ranges[0][0] == 0
    assert row_ranges[-1][1] == masks.shape[0]
    assert all(left[1] == right[0] for left, right in zip(row_ranges, row_ranges[1:]))
    expected_chunks = sum(int(item["chunk_count"]) for item in shard_events)
    assert len(shard_chunk_events) == expected_chunks
    assert all(int(item["chunk_ordinal"]) >= 1 for item in shard_chunk_events)
    assert all(int(item["chunk_ordinal"]) <= int(item["chunk_count"]) for item in shard_chunk_events)
    assert all(int(item["rows_completed_in_shard"]) > 0 for item in shard_chunk_events)
    assert all(int(item["shard_rows_total"]) > 0 for item in shard_chunk_events)
    assert all(float(item["duration_seconds"]) >= 0.0 for item in shard_chunk_events)


def test_finalize_subject_masks_process_shards_reuses_collection_worker_plan(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / "collection_analysis.zarr"
    _build_sharded_subject_mask_root(zarr_path)

    summary = mod.finalize_subject_masks(
        zarr_path,
        subject_shard_runs=["subject_masks_clip_b", "subject_masks_clip_a"],
        target_crop_run="crop_collection",
        refined_run="refined_subject_masks_collection_process_shards",
        components=["subject_body", "swim_bladder"],
        chunk_size=1,
        dense_mask_row_chunk=1,
        execution_backend="process_shards",
        num_workers=2,
    )
    mod.finalize_subject_masks(
        zarr_path,
        subject_shard_runs=["subject_masks_clip_b", "subject_masks_clip_a"],
        target_crop_run="crop_collection",
        refined_run="refined_subject_masks_collection_serial",
        components=["subject_body", "swim_bladder"],
        chunk_size=1,
        dense_mask_row_chunk=1,
        execution_backend="serial_driver",
    )

    plan = summary["collection_worker_index_plan"]
    assert plan["schema_id"] == mod._COLLECTION_WORKER_INDEX_PLAN_SCHEMA
    assert plan["global_identity_map_builds"] == 1
    assert plan["worker_identity_map_rebuilds"] == 0
    assert plan["row_count"] == 2
    root = zarr.open_group(str(zarr_path), mode="r")
    process_run = root["refined_subject_masks_runs/refined_subject_masks_collection_process_shards"]
    serial_run = root["refined_subject_masks_runs/refined_subject_masks_collection_serial"]
    np.testing.assert_array_equal(process_run["masks_roi"][:], serial_run["masks_roi"][:])
    provenance_plan = process_run.attrs["provenance"]["parameters"]["collection_worker_index_plan"]
    assert provenance_plan["worker_identity_map_rebuilds"] == 0


def test_collection_worker_initialization_benchmark_is_read_only(tmp_path: Path) -> None:
    zarr_path = tmp_path / "collection_init.zarr"
    root = _build_sharded_subject_mask_root(zarr_path)
    source_refined = root.create_group("refined_subject_masks_runs").create_group("failed_source")
    source_refined.attrs.update(
        {
            "source_subject_mask_shard_runs": [
                "subject_masks_clip_b",
                "subject_masks_clip_a",
            ],
            "source_crop_rebase_target_run": "crop_collection",
        }
    )
    output_path = tmp_path / "worker_init.json"

    summary = benchmark_collection_worker_initialization(
        zarr_path,
        source_refined_run="failed_source",
        num_workers=2,
        sample_rows=1,
        hold_seconds=0,
        output_json=output_path,
    )

    assert summary["worker_count"] == 2
    assert summary["collection_worker_plan"]["worker_identity_map_rebuilds"] == 0
    assert summary["worker_current_rss_sum_kib"] > 0
    assert len(summary["workers"]) == 2
    assert all(worker["sample_rows"] == 1 for worker in summary["workers"])
    assert output_path.is_file()
    reopened = zarr.open_group(str(zarr_path), mode="r")
    assert list(reopened["refined_subject_masks_runs"].group_keys()) == ["failed_source"]


def test_build_finalizer_ab_fixture_changes_only_probability_layout(tmp_path: Path) -> None:
    source_path = tmp_path / "source_collection.zarr"
    root = _build_sharded_subject_mask_root(source_path)
    keypoint_parent = root.create_group("refined_keypoints_runs")
    keypoint_parent.attrs["latest"] = "refined_keypoints_collection"
    keypoints = keypoint_parent.create_group("refined_keypoints_collection")
    keypoints.attrs["keypoint_labels"] = ["swim_bladder", "eye_left", "eye_right"]
    keypoints.create_array(
        "keypoints_roi",
        data=np.asarray(
            [
                [[5.0, 6.0], [2.0, 2.0], [7.0, 2.0]],
                [[5.0, 6.0], [2.0, 4.0], [7.0, 4.0]],
            ],
            dtype=np.float32,
        ),
        overwrite=True,
    )
    keypoints.create_array("detection_success", data=np.asarray([True, True]), overwrite=True)
    keypoints.create_array("source_crop_row_ids", data=np.asarray([0, 1], dtype=np.int64), overwrite=True)
    output_root = tmp_path / "fixture"

    summary = build_finalizer_ab_fixture(
        source_path,
        source_shard_run="subject_masks_clip_a",
        target_crop_run="crop_collection",
        assignment_keypoint_group="refined_keypoints_runs",
        assignment_keypoints_run="refined_keypoints_collection",
        output_root=output_root,
        inner_chunk_rows=1,
        shard_rows=2,
    )

    assert summary["row_count"] == 1
    assert summary["all_exact"] is True
    assert len({variant["destination_sha256"] for variant in summary["variants"]}) == 1
    regular = zarr.open_group(str(output_root / "regular.zarr"), mode="r")
    sharded = zarr.open_group(str(output_root / "shard_02048.zarr"), mode="r")
    regular_probs = regular["subject_mask_runs/subject_masks_finalizer_ab_fixture/mask_probs_roi"]
    sharded_probs = sharded["subject_mask_runs/subject_masks_finalizer_ab_fixture/mask_probs_roi"]
    assert regular_probs.shards is None
    assert sharded_probs.shards == (2, 1, 10, 10)
    np.testing.assert_array_equal(regular_probs[:], sharded_probs[:])
    assert regular["crop_runs/crop_finalizer_ab_fixture/frame_indices"].shape == (1,)
    assert regular["refined_keypoints_runs/refined_keypoints_finalizer_ab_fixture/keypoints_roi"].shape[0] == 1
    source_reopened = zarr.open_group(str(source_path), mode="r")
    assert "refined_subject_masks_runs" not in source_reopened


def test_finalize_subject_masks_process_shards_can_write_dense_plus_bitpacked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / "analysis.zarr"
    _build_probability_root(zarr_path)

    summary = mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_process_shards_bitpacked_001",
        chunk_size=1,
        execution_backend="process_shards",
        num_workers=2,
        mask_storage="dense_and_bitpacked",
    )

    assert summary["execution_backend"] == "process_shards"
    assert summary["mask_storage"] == "dense_and_bitpacked"
    assert summary["mask_bitpacked_summary"]["status"] == "written"
    assert summary["mask_bitpacked_summary"]["roundtrip_validation"]["status"] == "passed"
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_masks_smart_process_shards_bitpacked_001"]
    assert "masks_roi" in run
    assert "mask_bitpacked/masks_packed" in run
    assert run.attrs["mask_storage_encoding"] == "dense_uint8+bitpacked_binary_v1"
    assert run.attrs["masks_roi_materialized"] is True
    assert run.attrs["mask_bitpacked_materialized"] is True
    store = open_mask_store(run, prefer="bitpacked")
    assert store.read_dense().shape == (2, 4, 10, 10)
    phase_seconds = run.attrs["smart_finalizer_timing_summary"]["phase_seconds"]
    assert "write_bitpacked_mask_store" in phase_seconds


def test_worker_chunk_size_aligns_to_dense_mask_row_chunk() -> None:
    assert mod._worker_chunk_size_for_backend(
        120_221,
        256,
        "process_shards",
        dense_mask_row_chunk=256,
    ) == 256
    assert mod._worker_chunk_size_for_backend(
        120_221,
        256,
        "process_shards",
        dense_mask_row_chunk=512,
    ) == 512


def test_finalize_subject_masks_records_dense_mask_row_chunk(monkeypatch, tmp_path: Path) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / "analysis.zarr"
    _build_probability_root(zarr_path)

    summary = mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_dense_chunk_001",
        chunk_size=1,
        dense_mask_row_chunk=512,
    )

    assert summary["dense_mask_row_chunk"] == 2
    assert summary["dense_mask_storage_chunks"] == [2, 1, 10, 10]
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs/refined_subject_masks_dense_chunk_001"]
    assert tuple(int(v) for v in run["masks_roi"].chunks) == (2, 1, 10, 10)
    assert run.attrs["dense_mask_row_chunk"] == 2
    assert run.attrs["dense_mask_storage_chunks"] == [2, 1, 10, 10]
    provenance_parameters = run.attrs["provenance"]["parameters"]
    assert provenance_parameters["dense_mask_row_chunk"] == 2
    assert provenance_parameters["dense_mask_storage_chunks"] == [2, 1, 10, 10]


def test_refresh_refined_subject_mask_metrics_can_refresh_component_contours(monkeypatch, tmp_path: Path) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    zarr_path = tmp_path / "analysis.zarr"
    _build_probability_root(zarr_path)
    mod.finalize_subject_masks(
        zarr_path,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
        write_component_contours=True,
    )
    root = zarr.open_group(str(zarr_path), mode="a")
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    labels = list(run.attrs["mask_labels"])
    body_idx = labels.index("subject_body")
    original_len = np.asarray(run["components/subject_body/contours/len"][:], dtype=np.int32).copy()

    edited_body = np.zeros((10, 10), dtype=np.uint8)
    edited_body[1:4, 1:4] = 1
    run["masks_roi"][0, body_idx] = edited_body

    summary = mod.refresh_refined_subject_mask_metrics(
        zarr_path,
        refined_run="refined_subject_masks_smart_001",
        components=["subject_body"],
        chunk_size=1,
        metric_level="cheap",
        write_component_contours=True,
    )

    assert summary["write_component_contours"] is True
    assert summary["component_contours"][0]["component"] == "subject_body"
    assert summary["component_contours"][0]["status"] == "written"
    root = zarr.open_group(str(zarr_path), mode="r")
    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    refreshed_len = np.asarray(run["components/subject_body/contours/len"][:], dtype=np.int32)
    assert int(refreshed_len[0]) != int(original_len[0])
    assert int(refreshed_len[0]) > 0
    assert run.attrs["component_contours_status"] == "computed"


def test_finalize_subject_mask_run_dry_run_and_overwrite_guard(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    dry = mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
        dry_run=True,
    )

    assert dry["status"] == "planned"
    assert dry["mutates_archive"] is False
    assert dry["metric_level"] == "cheap"
    assert dry["write_eye_geometry"] is True
    assert dry["write_component_contours"] is True
    assert dry["write_sampled_component_contours"] is True
    assert dry["execution_backend"] == "serial_driver"
    assert dry["process_shard_execution_enabled"] is False
    assert "refined_subject_masks_runs" not in root

    mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
    )
    with pytest.raises(ValueError, match="already exists"):
        mod.finalize_subject_mask_run(
            root,
            subject_run="subject_probs_001",
            refined_run="refined_subject_masks_smart_001",
            chunk_size=1,
        )


def test_finalize_subject_mask_run_copies_instance_key_from_source(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()
    keys = np.asarray([11258999068426240, 22517998136852481], dtype=np.uint64)
    root["subject_mask_runs"]["subject_probs_001"].create_array(
        "instance_key", data=keys, overwrite=True
    )

    mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
    )

    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    assert "instance_key" in run
    copied = np.asarray(run["instance_key"][:], dtype=np.uint64)
    np.testing.assert_array_equal(copied, keys)


def test_finalize_subject_mask_run_omits_instance_key_for_legacy_source(monkeypatch) -> None:
    _patch_refined_subject_provenance(monkeypatch)
    root = _build_probability_root()

    mod.finalize_subject_mask_run(
        root,
        subject_run="subject_probs_001",
        refined_run="refined_subject_masks_smart_001",
        chunk_size=1,
    )

    run = root["refined_subject_masks_runs"]["refined_subject_masks_smart_001"]
    assert "instance_key" not in run
