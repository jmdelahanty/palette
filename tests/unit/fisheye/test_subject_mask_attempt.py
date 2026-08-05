from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import zarr

from fisheye.shared.subject_mask_attempt import (
    build_subject_mask_attempt,
    build_subject_mask_scientific_identity,
    resolve_subject_mask_attempt_lineage,
    validate_subject_mask_attempt,
    validate_subject_mask_scientific_identity,
)
from fisheye.shared.zarr_run_completion import RUN_COMPLETION_STATUS_ATTR
from fisheye.shared.subject_mask_worker_receipt import (
    REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
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
        science = build_subject_mask_scientific_identity(
            stage_kind="refined_subject_mask",
            model={"policy": "v1"},
            crop={"clip": worker_index},
            pixels={"source": "raw_masks"},
            row_identity={"rows": 2},
            inference_contract={"components": ["body", "eye"]},
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
        )
        run_path = f"refined_subject_masks_runs/clip_{index}"
        attempt = build_subject_mask_attempt(
            scientific_identity=science,
            run_path=run_path,
            attempt_id=f"00000000-0000-4000-8000-{index + 1:012d}",
        )
        local = {
            path: arrays[path][index : index + 1]
            if path != "available_channels"
            else arrays[path]
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
