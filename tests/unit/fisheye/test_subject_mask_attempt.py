from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import zarr

from fisheye.shared.subject_mask_attempt import (
    SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
    build_subject_mask_attempt,
    build_subject_mask_scientific_identity,
    resolve_subject_mask_attempt_lineage,
    validate_subject_mask_attempt,
    validate_subject_mask_scientific_identity,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from fisheye.shared.subject_mask_worker_receipt import (
    REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    _normalize_expected_work_units,
    build_recording_assignment_keypoint_collection,
    build_recording_subject_mask_source_receipt,
    build_subject_mask_worker_semantic_receipt,
    validate_recording_subject_mask_assembly_identity,
    validate_subject_mask_worker_semantic_receipt,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    subject_mask_array_unit_document,
    validate_subject_mask_source_validation_receipt,
)
from fisheye.shared.zarr.subject_mask_schema import (
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
)


def _science(*, pixels_sha256: str = "a" * 64) -> dict[str, object]:
    return build_subject_mask_scientific_identity(
        stage_kind="raw_subject_mask",
        model={"artifact_sha256": "b" * 64},
        crop={"run_id": "crop_v2"},
        pixels={"decoded_pixels_sha256": pixels_sha256},
        row_identity={"instance_key_sha256": "c" * 64},
        inference_contract={"label_schema_id": "subject_v1_union"},
        schema_version=1,
    )


def test_expected_work_units_retain_empty_frame_windows() -> None:
    units = [
        {
            "work_unit_id": "collection:clip_0",
            "work_unit_index": 0,
            "source_clip_id": "clip_0",
            "source_clip_index": 0,
            "frame_start": 0,
            "frame_stop": 1,
            "row_start": 0,
            "row_stop": 2,
        },
        {
            "work_unit_id": "collection:clip_empty",
            "work_unit_index": 1,
            "source_clip_id": "clip_empty",
            "source_clip_index": 1,
            "frame_start": 1,
            "frame_stop": 2,
            "row_start": 2,
            "row_stop": 2,
        },
        {
            "work_unit_id": "collection:clip_2",
            "work_unit_index": 2,
            "source_clip_id": "clip_2",
            "source_clip_index": 2,
            "frame_start": 2,
            "frame_stop": 4,
            "row_start": 2,
            "row_stop": 4,
        },
    ]

    assert _normalize_expected_work_units(units, n_frames=4, n_rois=4) == units
    with pytest.raises(ValueError, match="exactly and contiguously cover"):
        _normalize_expected_work_units(
            [units[0], {**units[2], "work_unit_index": 1}],
            n_frames=4,
            n_rois=4,
        )


def _raw_science_v2() -> dict[str, object]:
    rows = 2
    row_arrays = {
        "source_crop_row_ids": {
            "shape": [rows],
            "dtype": "int64",
            "sha256": "1" * 64,
        },
        "instance_key": {
            "shape": [rows],
            "dtype": "uint64",
            "sha256": "2" * 64,
        },
        "source_acquisition_frame_index": {
            "shape": [rows],
            "dtype": "int64",
            "sha256": "3" * 64,
        },
    }
    return build_subject_mask_scientific_identity(
        stage_kind="raw_subject_mask",
        model={
            "artifact_role": "subject_mask_checkpoint",
            "artifact_sha256": "4" * 64,
            "artifact_size_bytes": 1024,
            "registry_set_id": "models_v1",
            "registry_run_id": "run_001",
            "label_schema_id": "subject_masks_v1",
        },
        crop={
            "run_id": "crop_v2",
            "run_group_path": "crop_runs/crop_v2",
            "run_manifest": {
                "schema_id": "palette.crop.run_manifest",
                "schema_version": 2,
                "payload_digest": "5" * 64,
            },
            "storage_mode": "geometry_only",
            "roi_shape_hw": [8, 8],
            "roi_coordinates_full": {
                "shape": [rows, 2],
                "dtype": "int32",
                "sha256": "6" * 64,
            },
            "source_collection_id": "collection_001",
            "source_clip_id": "clip_000",
            "source_clip_index": 0,
            "source_work_unit_id": "collection_001:clip_000",
            "source_shard_id": "clip_000",
            "collection_partition_contract": None,
        },
        pixels={
            "profile": "palette.crop_pixel_work_package",
            "decoded_shape": [rows, 8, 8],
            "decoded_dtype": "uint8",
            "decoded_order": "C",
            "decoded_pixels_sha256": "7" * 64,
            "declared_pixels_sha256": "7" * 64,
            "cache_key": "cache_001",
            "pixel_materialization_id": "pixels_001",
            "pixel_contract": {"schema": "palette_roi_pixel_contract_v1"},
            "work_package_role": "complete_collection_partition",
        },
        row_identity={"row_count": rows, "arrays": row_arrays},
        inference_contract={
            "segmenter": "unet",
            "label_schema_id": "subject_masks_v1",
            "mask_labels": ["subject_body"],
            "model_input_transform": {
                "name": "identity",
                "native_shape_hw": [8, 8],
                "model_shape_hw": [8, 8],
                "pad_top": 0,
                "pad_bottom": 0,
                "pad_left": 0,
                "pad_right": 0,
                "coordinate_mapping": ("native_xy = model_xy - [pad_left, pad_top]"),
            },
            "probability_semantics": "sigmoid_multilabel_logits",
            "probability_dtype": "uint8",
            "probability_encoding": "linear_uint8_0_255",
            "mask_probability_threshold": 0.5,
            "overlap_policy": "independent_sigmoid",
        },
    )


def test_scientific_identity_is_exact_and_content_sensitive() -> None:
    first = _science()
    same = _science()
    changed = _science(pixels_sha256="d" * 64)

    assert first == same
    assert first["digest"] != changed["digest"]
    assert validate_subject_mask_scientific_identity(first) == ()

    tampered = deepcopy(first)
    tampered["payload"]["pixels"]["decoded_pixels_sha256"] = "e" * 64
    assert "scientific identity digest mismatch" in (
        validate_subject_mask_scientific_identity(tampered)
    )


def test_scientific_identity_v2_rejects_recomputed_nested_tampering() -> None:
    science = _raw_science_v2()
    assert science["schema_version"] == SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION
    assert validate_subject_mask_scientific_identity(science) == ()

    missing = deepcopy(science)
    del missing["payload"]["model"]["artifact_sha256"]
    missing["digest"] = canonical_json_sha256(missing["payload"])
    assert "raw scientific model fields are not exact" in (
        validate_subject_mask_scientific_identity(missing)
    )

    extra = deepcopy(science)
    extra["payload"]["pixels"]["uncontracted_backend"] = "cuda"
    extra["digest"] = canonical_json_sha256(extra["payload"])
    assert "raw scientific pixels fields are not exact" in (
        validate_subject_mask_scientific_identity(extra)
    )


def test_scientific_identity_v2_rejects_empty_nested_documents() -> None:
    with pytest.raises(ValueError, match="raw scientific model fields"):
        build_subject_mask_scientific_identity(
            stage_kind="raw_subject_mask",
            model={},
            crop={},
            pixels={},
            row_identity={},
            inference_contract={},
        )


def test_attempt_separates_execution_identity_from_scientific_identity() -> None:
    science = _science()
    first = build_subject_mask_attempt(
        scientific_identity=science,
        run_path="subject_mask_shard_runs/run_001",
        attempt_id="00000000-0000-4000-8000-000000000001",
    )
    retry = build_subject_mask_attempt(
        scientific_identity=science,
        run_path="subject_mask_shard_runs/run_002",
        attempt_id="00000000-0000-4000-8000-000000000002",
        retry_of_attempt_id="00000000-0000-4000-8000-000000000001",
        supersedes_run="run_000",
    )

    assert validate_subject_mask_attempt(first) == ()
    assert validate_subject_mask_attempt(retry) == ()
    assert first["payload"]["scientific_identity_digest"] == science["digest"]
    assert retry["payload"]["scientific_identity_digest"] == science["digest"]
    assert first["payload_digest"] != retry["payload_digest"]


def test_attempt_rejects_self_retry_and_non_run_paths() -> None:
    science = _science()
    attempt_id = "00000000-0000-4000-8000-000000000001"
    with pytest.raises(ValueError, match="cannot equal"):
        build_subject_mask_attempt(
            scientific_identity=science,
            run_path="subject_mask_runs/run_001",
            attempt_id=attempt_id,
            retry_of_attempt_id=attempt_id,
        )
    with pytest.raises(ValueError, match="family and run name"):
        build_subject_mask_attempt(
            scientific_identity=science,
            run_path="run_001",
        )


def test_attempt_lineage_binds_failed_retry_and_complete_supersession() -> None:
    parent = zarr.group().require_group("refined_subject_masks_runs")
    science = _science()
    failed = parent.create_group("failed_run")
    failed_attempt = build_subject_mask_attempt(
        scientific_identity=science,
        run_path="refined_subject_masks_runs/failed_run",
        attempt_id="00000000-0000-4000-8000-000000000001",
    )
    failed.attrs["subject_mask_scientific_identity"] = science
    failed.attrs["subject_mask_attempt"] = failed_attempt
    failed.attrs[RUN_COMPLETION_STATUS_ATTR] = "failed"

    complete = parent.create_group("complete_run")
    complete.attrs["subject_mask_scientific_identity"] = science
    complete.attrs["subject_mask_attempt"] = build_subject_mask_attempt(
        scientific_identity=science,
        run_path="refined_subject_masks_runs/complete_run",
        attempt_id="00000000-0000-4000-8000-000000000002",
    )
    complete.attrs[RUN_COMPLETION_STATUS_ATTR] = "complete"

    candidate = build_subject_mask_attempt(
        scientific_identity=science,
        run_path="refined_subject_masks_runs/candidate",
        attempt_id="00000000-0000-4000-8000-000000000003",
        retry_of_attempt_id="00000000-0000-4000-8000-000000000001",
        supersedes_run="complete_run",
    )
    evidence = resolve_subject_mask_attempt_lineage(
        parent=parent,
        current_run_name="candidate",
        scientific_identity=science,
        attempt=candidate,
        retry_of_attempt_id="00000000-0000-4000-8000-000000000001",
        supersedes_run="complete_run",
    )

    assert evidence["retry_of"]["run_name"] == "failed_run"
    assert evidence["supersedes"]["run_name"] == "complete_run"


def test_attempt_lineage_rejects_retry_with_changed_science() -> None:
    parent = zarr.group().require_group("subject_mask_runs")
    prior_science = _science()
    failed = parent.create_group("failed_run")
    failed.attrs["subject_mask_scientific_identity"] = prior_science
    failed.attrs["subject_mask_attempt"] = build_subject_mask_attempt(
        scientific_identity=prior_science,
        run_path="subject_mask_runs/failed_run",
        attempt_id="00000000-0000-4000-8000-000000000001",
    )
    failed.attrs[RUN_COMPLETION_STATUS_ATTR] = "failed"
    changed_science = _science(pixels_sha256="f" * 64)
    candidate = build_subject_mask_attempt(
        scientific_identity=changed_science,
        run_path="subject_mask_runs/candidate",
        attempt_id="00000000-0000-4000-8000-000000000002",
        retry_of_attempt_id="00000000-0000-4000-8000-000000000001",
    )

    with pytest.raises(ValueError, match="exact same scientific identity"):
        resolve_subject_mask_attempt_lineage(
            parent=parent,
            current_run_name="candidate",
            scientific_identity=changed_science,
            attempt=candidate,
            retry_of_attempt_id="00000000-0000-4000-8000-000000000001",
            supersedes_run=None,
        )


def test_worker_semantic_receipt_binds_exact_row_units() -> None:
    science = _science()
    attempt = build_subject_mask_attempt(
        scientific_identity=science,
        run_path="subject_mask_shard_runs/run_001",
        attempt_id="00000000-0000-4000-8000-000000000001",
    )
    arrays = {
        "mask_probs_roi": np.arange(4 * 2, dtype=np.uint8).reshape(4, 2),
        "metrics/prob_max": np.arange(4 * 2, dtype=np.float32).reshape(4, 2),
        "available_channels": np.ones((2,), dtype=bool),
    }
    paths = tuple(arrays)
    document = subject_mask_array_unit_document(arrays, paths, unit_rows=2)
    receipt = build_subject_mask_worker_semantic_receipt(
        stage_kind="raw_subject_mask",
        run_path="subject_mask_shard_runs/run_001",
        scientific_identity=science,
        attempt=attempt,
        scope={"clip_id": "clip_001"},
        row_count=4,
        array_document=document,
        required_paths=paths,
        roi_aligned_paths=("mask_probs_roi", "metrics/prob_max"),
    )

    assert (
        validate_subject_mask_worker_semantic_receipt(
            receipt,
            scientific_identity=science,
            attempt=attempt,
            required_paths=paths,
        )
        == receipt
    )

    tampered = deepcopy(receipt)
    tampered["payload"]["arrays"]["mask_probs_roi"]["units"][0]["sha256"] = "0" * 64
    tampered["payload"]["arrays"]["mask_probs_roi"]["units_digest"] = "0" * 64
    with pytest.raises(ValueError):
        validate_subject_mask_worker_semantic_receipt(
            tampered,
            scientific_identity=science,
            attempt=attempt,
            required_paths=paths,
        )


def test_recording_receipt_concatenates_real_worker_intervals() -> None:
    n_rows, n_channels, height, width = 4, 2, 2, 2
    dimensions = SubjectMaskDimensions(
        n_frames=5,
        n_rois=n_rows,
        n_channels=n_channels,
        roi_height=height,
        roi_width=width,
    )
    components = SubjectMaskComponentRegistry(("body", "eye"))
    masks = np.arange(n_rows * n_channels * height * width, dtype=np.uint8).reshape(
        n_rows, n_channels, height, width
    )
    present = np.ones((n_rows, n_channels), dtype=bool)
    arrays = {
        "source_crop_row_ids": np.arange(n_rows, dtype=np.int64),
        "instance_key": np.arange(10, 10 + n_rows, dtype=np.uint64),
        "source_acquisition_frame_index": np.asarray([0, 1, 3, 4], dtype=np.int64),
        "frame_row_offsets": np.asarray([0, 1, 2, 2, 3, 4], dtype=np.int64),
        "source_crop_xywh": np.ones((n_rows, 4), dtype=np.float32),
        "masks_roi": masks,
        "available_channels": np.ones((n_channels,), dtype=bool),
        "metrics/mask_present": present,
        "metrics/area_px": np.full((n_rows, n_channels), 4, dtype=np.float32),
        "metrics/centroid_xy": np.ones((n_rows, n_channels, 2), dtype=np.float32),
        "metrics/centroid_valid": present.copy(),
        "metrics/bbox_xyxy": np.ones((n_rows, n_channels, 4), dtype=np.float32),
        "metrics/bbox_valid": present.copy(),
    }
    workers = []
    for worker_index, start in enumerate((0, 2), start=1):
        stop = start + 2
        assignment = {
            "assignment_keypoints_run": f"keypoints_clip_{worker_index}",
            "assignment_keypoint_group": "refined_keypoints_runs",
            "assignment_keypoint_contract": (
                "subject_eyes_union_assignment_keypoints_v1"
            ),
            "assignment_keypoint_role": "eyes_union_lr_assignment",
            "assignment_keypoint_selection": "explicit_fixture",
            "assignment_keypoint_success_dataset": "usable_keypoints",
            "assignment_keypoint_row_identity": {
                "row_identity_check": "source_crop_row_ids_subset",
                "rows_checked": 2,
                "keypoint_has_source_crop_row_ids": True,
                "mask_has_source_crop_row_ids": True,
                "keypoint_rows_available": n_rows,
                "keypoint_rows_selected": 2,
                "keypoint_selection_min_row": start,
                "keypoint_selection_max_row": stop - 1,
            },
            "assignment_keypoint_row_identity_check": ("source_crop_row_ids_subset"),
        }
        science = build_subject_mask_scientific_identity(
            stage_kind="refined_subject_mask",
            model={"policy": "v1"},
            crop={"clip": worker_index},
            pixels={"source": "raw_masks"},
            row_identity={"rows": 2},
            inference_contract={
                "components": ["body", "eye"],
                "eye_assignment_contract": assignment,
            },
            schema_version=1,
        )
        attempt = build_subject_mask_attempt(
            scientific_identity=science,
            run_path=f"refined_subject_masks_runs/clip_{worker_index}",
            attempt_id=f"00000000-0000-4000-8000-{worker_index:012d}",
        )
        local_arrays = {
            path: (
                arrays[path][start:stop]
                if path != "available_channels"
                else arrays[path]
            )
            for path in REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS
        }
        receipt = build_subject_mask_worker_semantic_receipt(
            stage_kind="refined_subject_mask",
            run_path=f"refined_subject_masks_runs/clip_{worker_index}",
            scientific_identity=science,
            attempt=attempt,
            scope={"clip": worker_index},
            row_count=2,
            array_document=subject_mask_array_unit_document(
                local_arrays,
                REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
                unit_rows=1,
            ),
            required_paths=REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
            roi_aligned_paths=tuple(
                path
                for path in REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS
                if path != "available_channels"
            ),
        )
        workers.append(
            {
                "global_start_row": start,
                "scientific_identity": science,
                "attempt": attempt,
                "receipt": receipt,
            }
        )

    source_manifest, source_receipt = build_recording_subject_mask_source_receipt(
        kind="refined_dense_core",
        stage_kind="refined_subject_mask",
        source_run_path="refined_subject_masks_runs/recording",
        schema=REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
        arrays=arrays,
        dimensions=dimensions,
        components=components,
        threshold=None,
        workers=workers,
        identity_unit_rows=2,
    )
    assert source_receipt["payload"]["semantic_coverage"]["stop_row"] == 4
    assert source_receipt["payload"]["semantic_coverage"]["unit_count"] == 4
    assert source_receipt["schema_version"] == 2
    producer_evidence = source_receipt["payload"]["producer_evidence"]
    assert len(producer_evidence["workers"]) == 2
    assert producer_evidence["workers"][0]["scientific_identity"] == (
        workers[0]["scientific_identity"]
    )
    assert (
        validate_recording_subject_mask_assembly_identity(
            producer_evidence,
            kind="refined_dense_core",
            stage_kind="refined_subject_mask",
            source_run_path="refined_subject_masks_runs/recording",
            n_rois=4,
        )
        == producer_evidence
    )
    assignment_collection = build_recording_assignment_keypoint_collection(
        producer_evidence,
        source_run_path="refined_subject_masks_runs/recording",
        n_rois=4,
    )
    assert assignment_collection["mode"] == "exact_worker_partition"
    assert [
        worker["assignment"]["assignment_keypoints_run"]
        for worker in assignment_collection["workers"]
    ] == ["keypoints_clip_1", "keypoints_clip_2"]
    assert (
        validate_subject_mask_source_validation_receipt(
            source_receipt,
            kind="refined_dense_core",
            source_run_path="refined_subject_masks_runs/recording",
            source_manifest=source_manifest,
            schema=REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
            arrays=arrays,
            dimensions=dimensions,
            components=components,
            threshold=None,
        )
        == source_receipt
    )

    workers[1]["global_start_row"] = 3
    with pytest.raises(ValueError, match="ordered, contiguous"):
        build_recording_subject_mask_source_receipt(
            kind="refined_dense_core",
            stage_kind="refined_subject_mask",
            source_run_path="refined_subject_masks_runs/recording",
            schema=REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
            arrays=arrays,
            dimensions=dimensions,
            components=components,
            threshold=None,
            workers=workers,
        )


def test_recording_receipt_rejects_conflicting_worker_science() -> None:
    n_rows = 2
    dimensions = SubjectMaskDimensions(
        n_frames=2,
        n_rois=n_rows,
        n_channels=2,
        roi_height=2,
        roi_width=2,
    )
    components = SubjectMaskComponentRegistry(("body", "eye"))
    masks = np.ones((n_rows, 2, 2, 2), dtype=np.uint8)
    metrics = {
        "metrics/mask_present": np.ones((n_rows, 2), dtype=bool),
        "metrics/area_px": np.full((n_rows, 2), 4, dtype=np.float32),
        "metrics/centroid_xy": np.ones((n_rows, 2, 2), dtype=np.float32),
        "metrics/centroid_valid": np.ones((n_rows, 2), dtype=bool),
        "metrics/bbox_xyxy": np.ones((n_rows, 2, 4), dtype=np.float32),
        "metrics/bbox_valid": np.ones((n_rows, 2), dtype=bool),
    }
    arrays = {
        "source_crop_row_ids": np.arange(n_rows, dtype=np.int64),
        "instance_key": np.arange(10, 12, dtype=np.uint64),
        "source_acquisition_frame_index": np.arange(n_rows, dtype=np.int64),
        "frame_row_offsets": np.arange(3, dtype=np.int64),
        "source_crop_xywh": np.ones((n_rows, 4), dtype=np.float32),
        "masks_roi": masks,
        "available_channels": np.ones((2,), dtype=bool),
        **metrics,
    }
    workers = []
    for index, model in enumerate(("policy_a", "policy_b")):
        science = build_subject_mask_scientific_identity(
            stage_kind="refined_subject_mask",
            model={"policy": model},
            crop={"roi_shape_hw": [2, 2]},
            pixels={"source": "raw_masks"},
            row_identity={"rows": 1},
            inference_contract={"components": ["body", "eye"]},
            schema_version=1,
        )
        run_path = f"refined_subject_masks_runs/clip_{index}"
        attempt = build_subject_mask_attempt(
            scientific_identity=science,
            run_path=run_path,
            attempt_id=f"00000000-0000-4000-8000-{index + 1:012d}",
        )
        local = {
            path: (
                arrays[path][index : index + 1]
                if path != "available_channels"
                else arrays[path]
            )
            for path in REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS
        }
        receipt = build_subject_mask_worker_semantic_receipt(
            stage_kind="refined_subject_mask",
            run_path=run_path,
            scientific_identity=science,
            attempt=attempt,
            scope={"clip": index},
            row_count=1,
            array_document=subject_mask_array_unit_document(
                local,
                REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
                unit_rows=1,
            ),
            required_paths=REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
            roi_aligned_paths=tuple(
                path
                for path in REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS
                if path != "available_channels"
            ),
        )
        workers.append(
            {
                "global_start_row": index,
                "scientific_identity": science,
                "attempt": attempt,
                "receipt": receipt,
            }
        )

    with pytest.raises(ValueError, match="conflicting scientific authority"):
        build_recording_subject_mask_source_receipt(
            kind="refined_dense_core",
            stage_kind="refined_subject_mask",
            source_run_path="refined_subject_masks_runs/recording",
            schema=REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
            arrays=arrays,
            dimensions=dimensions,
            components=components,
            threshold=None,
            workers=workers,
        )
