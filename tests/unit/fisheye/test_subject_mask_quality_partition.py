from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_sha256,
)
from fisheye.shared.zarr.subject_mask_quality_partition import (
    SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_ID,
    SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_VERSION,
    build_subject_mask_quality_partition_assembly,
    compute_subject_mask_quality_partition,
    load_subject_mask_quality_partition_arrays,
    validate_subject_mask_quality_partition,
)
from fisheye.shared.zarr.subject_mask_quality_manifest import (
    validate_subject_mask_quality_run_manifest,
)
from fisheye.shared.zarr.subject_mask_quality_producer import (
    SUBJECT_V1_LR_COMPONENTS,
    prepare_in_memory_observation_local_subject_mask_quality,
)
from fisheye.shared.zarr.subject_mask_quality_publication import (
    publish_selector_ineligible_subject_mask_quality_snapshot,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_schema import SubjectMaskComponentRegistry
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    subject_mask_array_unit_document,
)


class _Run(dict[str, np.ndarray]):
    def __init__(self, arrays: dict[str, np.ndarray], *, path: str) -> None:
        super().__init__(arrays)
        self.path = path
        self.attrs = {"mask_labels": list(SUBJECT_V1_LR_COMPONENTS)}


def _masks(rows: int = 4) -> np.ndarray:
    masks = np.zeros((rows, 4, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[:, 1, 2, 2] = 1
    masks[:, 2, 2, 5] = 1
    masks[:, 3, 5, 3] = 1
    if rows > 1:
        masks[1, 1, 0, 0] = 1
    return masks


def _worker_receipt(run_path: str, masks: np.ndarray) -> dict[str, object]:
    dense = subject_mask_array_unit_document(
        {"masks_roi": masks}, ("masks_roi",), unit_rows=2
    )["masks_roi"]
    payload = {
        "stage_kind": "refined_subject_mask",
        "run_path": run_path,
        "arrays": {"masks_roi": dense},
    }
    return {
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def _source_fixture() -> (
    tuple[dict[str, np.ndarray], dict[str, object], SubjectMaskQualitySourceReference]
):
    masks = _masks()
    arrays = {
        "masks_roi": masks,
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "source_acquisition_frame_index": np.asarray([0, 0, 2, 3], dtype=np.int64),
        "frame_row_offsets": np.asarray([0, 2, 2, 3, 4], dtype=np.int64),
        "available_channels": np.ones((4,), dtype=bool),
    }
    manifest: dict[str, object] = {
        "schema_id": "palette.refined_subject_mask.partition_test",
        "schema_version": 1,
    }
    components = SubjectMaskComponentRegistry(SUBJECT_V1_LR_COMPONENTS)
    source = SubjectMaskQualitySourceReference(
        run_name="refined_subject_masks_001",
        manifest_digest=canonical_json_sha256(manifest),
        dense_array_values_sha256=sha256_array(masks),
        component_registry_digest=canonical_json_sha256(components.as_manifest()),
        source_array_values_sha256={
            path: sha256_array(values) for path, values in arrays.items()
        },
    )
    return arrays, manifest, source


def test_quality_partition_matches_serial_values_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    arrays, _manifest, source = _source_fixture()
    run_path = "refined_subject_masks_runs/refined_worker_000"
    run = _Run(
        {
            "masks_roi": arrays["masks_roi"],
            "instance_key": arrays["instance_key"],
            "available_channels": arrays["available_channels"],
        },
        path=run_path,
    )
    worker_receipt = _worker_receipt(run_path, arrays["masks_roi"])
    destination = tmp_path / "partition"
    receipt = compute_subject_mask_quality_partition(
        run,
        source_acquisition_frame_index=arrays["source_acquisition_frame_index"],
        global_start_row=0,
        global_frame_start=0,
        global_frame_stop=4,
        work_unit_id="fixture:window_000",
        work_unit_index=0,
        source_worker_receipt=worker_receipt,
        producer_commit="a" * 40,
        destination=destination,
        compute_workers=2,
        source_compute_block_bytes=512,
        receipt_unit_rows=1024,
    )

    assert (
        validate_subject_mask_quality_partition(
            destination,
            source_worker_receipt=worker_receipt,
        )["payload_digest"]
        == receipt["payload_digest"]
    )
    expected = prepare_in_memory_observation_local_subject_mask_quality(
        arrays,
        n_frames=4,
        components=SubjectMaskComponentRegistry(SUBJECT_V1_LR_COMPONENTS),
        source=source,
    )
    observed = load_subject_mask_quality_partition_arrays(destination)
    for path, values in observed.items():
        wanted = np.asarray(expected.arrays[path])
        if np.issubdtype(values.dtype, np.floating):
            np.testing.assert_allclose(values, wanted, equal_nan=True)
        else:
            np.testing.assert_array_equal(values, wanted)

    changed_path = destination / "arrays" / "observation_quality_flags.npy"
    changed = np.load(changed_path)
    changed[0] ^= np.uint16(1)
    with changed_path.open("wb") as handle:
        np.save(handle, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="decoded arrays differ"):
        validate_subject_mask_quality_partition(
            destination,
            source_worker_receipt=worker_receipt,
        )


def test_recording_publication_adopts_receipt_bound_quality_partition(
    tmp_path: Path,
) -> None:
    arrays, source_manifest, source = _source_fixture()
    run_path = "refined_subject_masks_runs/refined_worker_000"
    run = _Run(
        {
            "masks_roi": arrays["masks_roi"],
            "instance_key": arrays["instance_key"],
            "available_channels": arrays["available_channels"],
        },
        path=run_path,
    )
    worker_receipt = _worker_receipt(run_path, arrays["masks_roi"])
    partition = tmp_path / "partition"
    receipt = compute_subject_mask_quality_partition(
        run,
        source_acquisition_frame_index=arrays["source_acquisition_frame_index"],
        global_start_row=0,
        global_frame_start=0,
        global_frame_stop=4,
        work_unit_id="fixture:window_000",
        work_unit_index=0,
        source_worker_receipt=worker_receipt,
        producer_commit="b" * 40,
        destination=partition,
        compute_workers=2,
        source_compute_block_bytes=512,
    )
    assembly = build_subject_mask_quality_partition_assembly(
        (receipt,), n_rois=4, producer_commit="b" * 40
    )
    precomputed = load_subject_mask_quality_partition_arrays(partition)
    shadow_root = tmp_path / "quality"
    publication = publish_selector_ineligible_subject_mask_quality_snapshot(
        arrays,
        n_frames=4,
        components=SubjectMaskComponentRegistry(SUBJECT_V1_LR_COMPONENTS),
        source=source,
        source_manifest=source_manifest,
        destination=shadow_root / "adopted.zarr",
        run_id="quality_adopted_001",
        shadow_root=shadow_root,
        scratch_root=tmp_path / "scratch",
        source_compute_block_bytes=512,
        precomputed_arrays=precomputed,
        worker_assembly=assembly,
        created_by="pytest",
    )

    assert publication.write_receipt["source_mode"] == (
        "receipt_bound_quality_partitions"
    )
    assert publication.write_receipt["worker_assembly"] == assembly
    assert publication.write_receipt["source_compute_workers_effective"] == 1
    assert publication.phase_seconds["ordered_source_verification"] >= 0.0
    quality = zarr.open_group(
        str(publication.output_path / "subject_mask_quality_runs" / publication.run_id),
        mode="r",
        use_consolidated=False,
    )
    expected = prepare_in_memory_observation_local_subject_mask_quality(
        arrays,
        n_frames=4,
        components=SubjectMaskComponentRegistry(SUBJECT_V1_LR_COMPONENTS),
        source=source,
    )
    for path in SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths:
        observed = np.asarray(quality[path][...])
        wanted = np.asarray(expected.arrays[path])
        if np.issubdtype(observed.dtype, np.floating):
            np.testing.assert_allclose(observed, wanted, equal_nan=True)
        else:
            np.testing.assert_array_equal(observed, wanted)

    tampered = copy.deepcopy(publication.manifest)
    tampered_assembly = tampered["payload"]["write_receipt"]["worker_assembly"]
    tampered_assembly["payload"]["workers"][0]["work_unit"]["global_row_interval"][
        "stop_row"
    ] = 3
    tampered_assembly["payload"]["workers_digest"] = canonical_json_sha256(
        tampered_assembly["payload"]["workers"]
    )
    tampered_assembly["payload_digest"] = canonical_json_sha256(
        tampered_assembly["payload"]
    )
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert any(
        "worker assembly is invalid" in error
        for error in validate_subject_mask_quality_run_manifest(tampered)
    )


def test_quality_partition_assembly_rejects_gaps() -> None:
    payload = {
        "kind": "subject_mask_quality_observation_partition",
        "work_unit": {
            "work_unit_id": "fixture:1",
            "work_unit_index": 1,
            "global_frame_interval": {"start_frame": 2, "stop_frame": 4},
            "global_row_interval": {"start_row": 2, "stop_row": 4},
        },
        "local_row_count": 2,
        "source_run_path": "refined_subject_masks_runs/worker_1",
        "source_dense_worker": {},
        "component_registry": {},
        "quality_profile": {},
        "quality_policy": {},
        "arrays": {},
        "array_document_digest": "0" * 64,
        "execution": {},
        "producer_commit": "c" * 40,
    }
    receipt = {
        "schema_id": SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_QUALITY_PARTITION_RECEIPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    with pytest.raises(ValueError, match="gap, overlap, or reordering"):
        build_subject_mask_quality_partition_assembly(
            (receipt,), n_rois=4, producer_commit="c" * 40
        )
