from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.instance_keys import mint_manual_curation_instance_keys
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.refined_detection_delta import (
    REFINED_DETECTION_DELTA_ARRAYS,
    REFINED_DETECTION_DELTA_OPERATION_CODE_MAP,
    RefinedDetectionDeltaBatch,
)
from fisheye.shared.zarr.refined_detection_delta_storage import (
    REFINED_DETECTION_DELTA_MAX_EVENTS_PER_PARTITION,
    REFINED_DETECTION_DELTA_PARENT,
    RefinedDetectionDeltaLineageBinding,
    RefinedDetectionDeltaStorageError,
    create_refined_detection_delta_generation,
    create_refined_detection_delta_lineage,
    freeze_refined_detection_delta_generation,
    read_frozen_refined_detection_delta_generation,
    read_refined_detection_delta_partition,
    refined_detection_delta_array_digest,
    rollover_refined_detection_delta_generation,
    write_refined_detection_delta_partition,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionDimensions,
)


LINEAGE_ID = "22222222-2222-4222-8222-222222222222"
BASE_SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"
BASE_MANIFEST_DIGEST = "a" * 64
BASE_LOGICAL_CONTENT_DIGEST = "b" * 64
RECORDING_IDENTITY = "delta_storage_recording"
CREATED_AT = "2026-07-27T12:00:00+00:00"


def _root(path: Path) -> zarr.Group:
    return zarr.open_group(str(path / "delta.zarr"), mode="w", zarr_format=3)


def _binding() -> RefinedDetectionDeltaLineageBinding:
    return RefinedDetectionDeltaLineageBinding(
        delta_lineage_id=LINEAGE_ID,
        base_run_path="refined_detect_runs/base_snapshot",
        base_snapshot_id=BASE_SNAPSHOT_ID,
        base_manifest_digest=BASE_MANIFEST_DIGEST,
        base_logical_content_digest=BASE_LOGICAL_CONTENT_DIGEST,
        recording_identity=RECORDING_IDENTITY,
        base_next_refined_row_id=0,
    )


def _manual_key(*, row_id: int, frame: int, bbox: list[float], class_id: int) -> int:
    return int(
        mint_manual_curation_instance_keys(
            recording_identity=RECORDING_IDENTITY,
            refined_row_ids=np.asarray([row_id], dtype=np.int64),
            frame_indices=np.asarray([frame], dtype=np.int32),
            bbox_norm_coords=np.asarray([bbox], dtype=np.float32),
            class_ids=np.asarray([class_id], dtype=np.int32),
        )[0]
    )


def _add_batch(
    *,
    partition_id: str = "partition_0001",
    sequence: int = 1,
    reason_code: int = 1,
    generation_ordinal: int = 0,
    row_id: int = 0,
) -> RefinedDetectionDeltaBatch:
    bbox = [0.5, 0.5, 0.2, 0.2]
    event = {
        "event_sequence": sequence,
        "expected_previous_event_sequence": 0,
        "operation_codes": REFINED_DETECTION_DELTA_OPERATION_CODE_MAP["add_instance"],
        "instance_key": _manual_key(
            row_id=row_id,
            frame=2,
            bbox=bbox,
            class_id=4,
        ),
        "refined_row_ids": row_id,
        "row_index_hint": -1,
        "timestamp_ns": 100,
        "reason_codes": reason_code,
        "payload_valid": True,
        "frame_indices": 2,
        "source_acquisition_frame_index": 2,
        "bbox_norm_coords": bbox,
        "scores": 0.0,
        "score_valid": False,
        "class_ids": 4,
        "source_kind_codes": SOURCE_KIND_CODE_MAP["manual"],
        "manual_edit_flags": True,
        "source_detect_row_index": -1,
    }
    arrays = {
        declaration.name: np.asarray(
            [event[declaration.name]],
            dtype=np.dtype(declaration.dtype),
        ).reshape(1, *declaration.trailing_shape)
        for declaration in REFINED_DETECTION_DELTA_ARRAYS
    }
    return RefinedDetectionDeltaBatch(
        delta_lineage_id=LINEAGE_ID,
        base_snapshot_id=BASE_SNAPSHOT_ID,
        base_manifest_digest=BASE_MANIFEST_DIGEST,
        generation_ordinal=generation_ordinal,
        partition_id=partition_id,
        actor_id="reviewer@example.org",
        reason_code_map={0: "none", 1: "missed_detection", 2: "alternate_reason"},
        arrays=arrays,
    )


def _empty_base() -> tuple[RefinedDetectionDimensions, dict[str, np.ndarray]]:
    dimensions = RefinedDetectionDimensions(
        n_frames=4,
        n_instances=0,
        n_source_detections=0,
        source_width=100,
        source_height=80,
    )
    concrete = dimensions.contract_dimensions
    arrays: dict[str, np.ndarray] = {}
    for binding in REFINED_DETECTION_SCHEMA_V1.bindings_for(dimensions):
        contract = REFINED_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        shape = tuple(
            dimension if isinstance(dimension, int) else concrete[dimension]
            for dimension in contract.shape_template
        )
        arrays[binding.path] = np.zeros(
            shape,
            dtype=np.dtype(contract.dtype.numpy_dtype),
        )
    assert REFINED_DETECTION_SCHEMA_V1.validate(arrays, dimensions=dimensions) == ()
    return dimensions, arrays


def _create(root: zarr.Group) -> None:
    create_refined_detection_delta_lineage(
        root,
        binding=_binding(),
        created_by="reviewer@example.org",
        created_at_utc=CREATED_AT,
    )


def test_persist_freeze_read_and_resolve_manual_add(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _create(root)
    batch = _add_batch()

    first = write_refined_detection_delta_partition(
        root,
        batch=batch,
        created_at_utc=CREATED_AT,
    )
    retry = write_refined_detection_delta_partition(
        root,
        batch=batch,
        created_at_utc="2026-07-27T13:00:00+00:00",
    )
    assert retry["payload_digest"] == first["payload_digest"]

    partition = root[
        f"{REFINED_DETECTION_DELTA_PARENT}/{LINEAGE_ID}/generations/"
        "generation_00000000000000000000/partitions/partition_0001"
    ]
    for declaration in REFINED_DETECTION_DELTA_ARRAYS:
        array = partition[declaration.name]
        assert array.chunks == batch.arrays[declaration.name].shape
        assert array.shards is None

    freeze_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=0,
        frozen_by="compactor",
        frozen_at_utc="2026-07-27T14:00:00+00:00",
    )
    frozen = read_frozen_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=0,
    )
    assert len(frozen.batches) == 1
    assert frozen.binding == _binding()
    post_freeze_retry = write_refined_detection_delta_partition(
        root,
        batch=batch,
        created_at_utc="2026-07-27T15:00:00+00:00",
    )
    assert post_freeze_retry["payload_digest"] == first["payload_digest"]

    dimensions, base_arrays = _empty_base()
    result = frozen.resolve(
        base_dimensions=dimensions,
        base_arrays=base_arrays,
        base_instance_reason_codes={0: "none"},
        base_source_reason_codes={0: "none"},
    )
    assert result.arrays["instances/frame_indices"].tolist() == [2]
    assert result.arrays["instances/frame_row_offsets"].tolist() == [0, 0, 0, 1, 1]
    assert result.arrays["instances/instance_key"].tolist() == [
        int(batch.arrays["instance_key"][0])
    ]

    with pytest.raises(RefinedDetectionDeltaStorageError, match="not open"):
        write_refined_detection_delta_partition(
            root,
            batch=_add_batch(partition_id="late", sequence=2),
            created_at_utc=CREATED_AT,
        )


def test_partition_retry_with_different_event_content_fails(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _create(root)
    write_refined_detection_delta_partition(
        root,
        batch=_add_batch(reason_code=1),
        created_at_utc=CREATED_AT,
    )

    with pytest.raises(RefinedDetectionDeltaStorageError, match="conflicts"):
        write_refined_detection_delta_partition(
            root,
            batch=_add_batch(reason_code=2),
            created_at_utc=CREATED_AT,
        )


def test_partition_array_tampering_fails_before_resolution(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _create(root)
    batch = _add_batch()
    write_refined_detection_delta_partition(
        root,
        batch=batch,
        created_at_utc=CREATED_AT,
    )
    partition = root[
        f"{REFINED_DETECTION_DELTA_PARENT}/{LINEAGE_ID}/generations/"
        "generation_00000000000000000000/partitions/partition_0001"
    ]
    partition["class_ids"][0] = np.int32(5)

    with pytest.raises(
        RefinedDetectionDeltaStorageError,
        match="reconstructed content or storage policy",
    ):
        read_refined_detection_delta_partition(
            root,
            delta_lineage_id=LINEAGE_ID,
            generation_ordinal=0,
            partition_id="partition_0001",
        )


def test_recomputed_manifest_tampering_still_fails_exact_reconstruction(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    _create(root)
    write_refined_detection_delta_partition(
        root,
        batch=_add_batch(),
        created_at_utc=CREATED_AT,
    )
    partition = root[
        f"{REFINED_DETECTION_DELTA_PARENT}/{LINEAGE_ID}/generations/"
        "generation_00000000000000000000/partitions/partition_0001"
    ]
    manifest = copy.deepcopy(dict(partition.attrs["partition_manifest"]))
    manifest["payload"]["storage_profile"]["profile_id"] = "forged_profile"
    manifest["payload_digest"] = canonical_json_sha256(manifest["payload"])
    partition.attrs["partition_manifest"] = manifest

    with pytest.raises(
        RefinedDetectionDeltaStorageError,
        match="reconstructed content or storage policy",
    ):
        read_refined_detection_delta_partition(
            root,
            delta_lineage_id=LINEAGE_ID,
            generation_ordinal=0,
            partition_id="partition_0001",
        )


def test_duplicate_event_sequence_is_rejected_across_partitions(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _create(root)
    write_refined_detection_delta_partition(
        root,
        batch=_add_batch(partition_id="partition_a", sequence=1),
        created_at_utc=CREATED_AT,
    )

    with pytest.raises(RefinedDetectionDeltaStorageError, match="collides"):
        write_refined_detection_delta_partition(
            root,
            batch=_add_batch(partition_id="partition_b", sequence=1),
            created_at_utc=CREATED_AT,
        )


def test_generation_chain_requires_frozen_predecessor_and_advancing_sequences(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    _create(root)
    with pytest.raises(RefinedDetectionDeltaStorageError, match="Only frozen"):
        create_refined_detection_delta_generation(
            root,
            delta_lineage_id=LINEAGE_ID,
            generation_ordinal=1,
            created_by="reviewer@example.org",
            created_at_utc=CREATED_AT,
        )

    write_refined_detection_delta_partition(
        root,
        batch=_add_batch(sequence=10),
        created_at_utc=CREATED_AT,
    )
    freeze_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=0,
        frozen_by="compactor",
        frozen_at_utc="2026-07-27T14:00:00+00:00",
    )
    create_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=1,
        created_by="reviewer@example.org",
        created_at_utc="2026-07-27T15:00:00+00:00",
    )
    with pytest.raises(RefinedDetectionDeltaStorageError, match="does not advance"):
        write_refined_detection_delta_partition(
            root,
            batch=_add_batch(
                partition_id="stale_sequence",
                sequence=10,
                generation_ordinal=1,
                row_id=1,
            ),
            created_at_utc=CREATED_AT,
        )

    write_refined_detection_delta_partition(
        root,
        batch=_add_batch(
            partition_id="generation_1_event",
            sequence=11,
            generation_ordinal=1,
            row_id=1,
        ),
        created_at_utc=CREATED_AT,
    )
    freeze_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=1,
        frozen_by="compactor",
        frozen_at_utc="2026-07-27T16:00:00+00:00",
    )
    frozen = read_frozen_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=1,
    )
    payload = frozen.generation_manifest["payload"]
    assert payload["previous_generation_ordinal"] == 0
    assert payload["minimum_event_sequence_exclusive"] == 10
    assert [batch.generation_ordinal for batch in frozen.batches] == [0, 1]
    dimensions, base_arrays = _empty_base()
    result = frozen.resolve(
        base_dimensions=dimensions,
        base_arrays=base_arrays,
        base_instance_reason_codes={0: "none"},
        base_source_reason_codes={0: "none"},
    )
    assert result.dimensions.n_instances == 2
    assert result.next_refined_row_id == 2
    assert result.report["generation_ordinals"] == [0, 1]


def test_generation_rollover_is_retry_safe_and_opens_before_compaction(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    _create(root)
    write_refined_detection_delta_partition(
        root,
        batch=_add_batch(sequence=10),
        created_at_utc=CREATED_AT,
    )

    first = rollover_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=0,
        next_generation_ordinal=1,
        actor_id="compactor",
        frozen_at_utc="2026-07-27T14:00:00+00:00",
        next_created_at_utc="2026-07-27T14:00:01+00:00",
    )
    retry = rollover_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=0,
        next_generation_ordinal=1,
        actor_id="compactor",
        frozen_at_utc="2026-07-27T14:00:00+00:00",
        next_created_at_utc="2026-07-27T14:00:01+00:00",
    )

    assert retry == first
    assert first["heavy_compaction_may_begin"] is True
    write_refined_detection_delta_partition(
        root,
        batch=_add_batch(
            partition_id="partition_next",
            sequence=11,
            generation_ordinal=1,
            row_id=1,
        ),
        created_at_utc="2026-07-27T15:00:00+00:00",
    )


def test_frozen_generation_rejects_membership_added_after_freeze(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    _create(root)
    write_refined_detection_delta_partition(
        root,
        batch=_add_batch(),
        created_at_utc=CREATED_AT,
    )
    freeze_refined_detection_delta_generation(
        root,
        delta_lineage_id=LINEAGE_ID,
        generation_ordinal=0,
        frozen_by="compactor",
        frozen_at_utc="2026-07-27T14:00:00+00:00",
    )
    generation = root[
        f"{REFINED_DETECTION_DELTA_PARENT}/{LINEAGE_ID}/generations/"
        "generation_00000000000000000000"
    ]
    generation["partitions"].create_group("injected_partition")

    with pytest.raises(
        RefinedDetectionDeltaStorageError,
        match="membership differs",
    ):
        read_frozen_refined_detection_delta_generation(
            root,
            delta_lineage_id=LINEAGE_ID,
            generation_ordinal=0,
        )


def test_array_digest_binds_name_dtype_shape_and_values() -> None:
    values = np.asarray([1, 2], dtype=np.uint64)
    digest = refined_detection_delta_array_digest("event_sequence", values)
    assert digest == refined_detection_delta_array_digest(
        "event_sequence",
        values.copy(),
    )
    assert digest != refined_detection_delta_array_digest(
        "event_sequence",
        np.asarray([1, 3], dtype=np.uint64),
    )


def test_partition_event_limit_is_frozen() -> None:
    assert REFINED_DETECTION_DELTA_MAX_EVENTS_PER_PARTITION == 65_536


def test_lineage_creation_rejects_zarr_v2_before_writing(tmp_path: Path) -> None:
    root = zarr.open_group(str(tmp_path / "legacy.zarr"), mode="w", zarr_format=2)

    with pytest.raises(RefinedDetectionDeltaStorageError, match="Zarr v3"):
        create_refined_detection_delta_lineage(
            root,
            binding=_binding(),
            created_by="reviewer@example.org",
            created_at_utc=CREATED_AT,
        )
    assert REFINED_DETECTION_DELTA_PARENT not in root
