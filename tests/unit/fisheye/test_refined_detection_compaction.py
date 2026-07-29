from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from fisheye.shared.instance_keys import (
    mint_detection_instance_keys,
    mint_manual_curation_instance_keys,
)
from fisheye.shared.zarr.detection_schema import (
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.refined_detection_compaction import (
    REFINED_DETECTION_COMPACTION_RECEIPT_NAME,
    RefinedDetectionCompactionError,
    compact_frozen_refined_detection_delta_generation,
    validate_refined_detection_compaction_receipt,
)
from fisheye.shared.zarr.refined_detection_delta import (
    REFINED_DETECTION_DELTA_ARRAYS,
    REFINED_DETECTION_DELTA_OPERATION_CODE_MAP,
    RefinedDetectionDeltaBatch,
)
from fisheye.shared.zarr.refined_detection_delta_storage import (
    RefinedDetectionDeltaLineageBinding,
    create_refined_detection_delta_lineage,
    freeze_refined_detection_delta_generation,
    write_refined_detection_delta_partition,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceIdentity,
)
from fisheye.shared.zarr.refined_detection_schema import SOURCE_KIND_CODE_MAP
from fisheye.shared.zarr.refined_detection_snapshot import (
    publish_selector_ineligible_refined_detection_snapshot,
    refined_detection_logical_hashes,
    require_safe_refined_detection_snapshot_destination,
)
from fisheye.shared.zarr.refined_detection_transition import (
    build_accept_all_refined_detection_root,
)


RECORDING_IDENTITY = "local_compaction_multi_subject"
BASE_SNAPSHOT_ID = "11111111-1111-4111-8111-111111111111"
SUCCESSOR_SNAPSHOT_ID = "22222222-2222-4222-8222-222222222222"
SNAPSHOT_LINEAGE_ID = "33333333-3333-4333-8333-333333333333"
DELTA_LINEAGE_ID = "44444444-4444-4444-8444-444444444444"
CREATED_AT = "2026-07-27T12:00:00+00:00"


def _canonical_transition():
    dimensions = CanonicalDetectionDimensions(
        n_frames=4,
        n_instances=3,
        source_width=100,
        source_height=80,
    )
    frames = np.asarray([0, 0, 2], dtype=np.int32)
    boxes = np.asarray(
        [
            [0.20, 0.20, 0.10, 0.10],
            [0.70, 0.20, 0.10, 0.10],
            [0.50, 0.70, 0.20, 0.10],
        ],
        dtype=np.float32,
    )
    classes = np.asarray([1, 2, 1], dtype=np.int32)
    bbox_img, centers = derive_canonical_detection_geometry(
        boxes,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    canonical = {
        "instances/frame_indices": frames,
        "instances/source_acquisition_frame_index": frames.astype(np.int64),
        "instances/instance_key": mint_detection_instance_keys(
            recording_identity=RECORDING_IDENTITY,
            frame_indices=frames,
            bbox_norm_coords=boxes,
            class_ids=classes,
        ),
        "instances/bbox_norm_coords": boxes,
        "instances/bbox_img_xyxy": bbox_img,
        "instances/centers_img_xy": centers,
        "instances/scores": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        "instances/class_ids": classes,
        "instances/frame_row_offsets": np.asarray([0, 2, 2, 3, 3], dtype=np.int64),
    }
    return build_accept_all_refined_detection_root(
        canonical,
        dimensions=dimensions,
        recording_identity=RECORDING_IDENTITY,
    )


def _base_publication(tmp_path: Path, *, coordinate_catalog: bool = False):
    transition = _canonical_transition()
    root = tmp_path / "snapshots"
    return publish_selector_ineligible_refined_detection_snapshot(
        dimensions=transition.dimensions,
        arrays=transition.arrays,
        instance_reason_codes=transition.instance_reason_codes,
        source_reason_codes=transition.source_reason_codes,
        destination=root / "base.zarr",
        run_id="refined_base",
        lineage=RefinedDetectionSnapshotLineage(
            lineage_id=SNAPSHOT_LINEAGE_ID,
            snapshot_id=BASE_SNAPSHOT_ID,
            recording_identity=RECORDING_IDENTITY,
            next_refined_row_id=3,
        ),
        source=RefinedDetectionSourceIdentity(
            run_id="detect_source",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        created_by="test",
        publication_kind="test_root",
        safe_root=root,
        coordinate_catalog=coordinate_catalog,
    )


def _manual_add_batch(*, base_manifest_digest: str) -> RefinedDetectionDeltaBatch:
    bbox = np.asarray([0.50, 0.45, 0.15, 0.15], dtype=np.float32)
    row_id = 3
    instance_key = int(
        mint_manual_curation_instance_keys(
            recording_identity=RECORDING_IDENTITY,
            refined_row_ids=np.asarray([row_id], dtype=np.int64),
            frame_indices=np.asarray([1], dtype=np.int32),
            bbox_norm_coords=bbox.reshape(1, 4),
            class_ids=np.asarray([4], dtype=np.int32),
        )[0]
    )
    event = {
        "event_sequence": 1,
        "expected_previous_event_sequence": 0,
        "operation_codes": REFINED_DETECTION_DELTA_OPERATION_CODE_MAP["add_instance"],
        "instance_key": instance_key,
        "refined_row_ids": row_id,
        "row_index_hint": -1,
        "timestamp_ns": 1,
        "reason_codes": 1,
        "payload_valid": True,
        "frame_indices": 1,
        "source_acquisition_frame_index": 1,
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
        delta_lineage_id=DELTA_LINEAGE_ID,
        base_snapshot_id=BASE_SNAPSHOT_ID,
        base_manifest_digest=base_manifest_digest,
        generation_ordinal=0,
        partition_id="partition_0001",
        actor_id="reviewer@example.org",
        reason_code_map={0: "none", 1: "missed_detection"},
        arrays=arrays,
    )


def _frozen_delta(tmp_path: Path, base):
    root = zarr.open_group(
        str(tmp_path / "delta.zarr"),
        mode="w",
        zarr_format=3,
    )
    create_refined_detection_delta_lineage(
        root,
        binding=RefinedDetectionDeltaLineageBinding(
            delta_lineage_id=DELTA_LINEAGE_ID,
            base_run_path="refined_detect_runs/refined_base",
            base_snapshot_id=BASE_SNAPSHOT_ID,
            base_manifest_digest=str(base.manifest["payload_digest"]),
            base_logical_content_digest=str(base.receipt["logical_content_digest"]),
            recording_identity=RECORDING_IDENTITY,
            base_next_refined_row_id=3,
        ),
        created_by="reviewer@example.org",
        created_at_utc=CREATED_AT,
    )
    write_refined_detection_delta_partition(
        root,
        batch=_manual_add_batch(
            base_manifest_digest=str(base.manifest["payload_digest"]),
        ),
        created_at_utc=CREATED_AT,
    )
    freeze_refined_detection_delta_generation(
        root,
        delta_lineage_id=DELTA_LINEAGE_ID,
        generation_ordinal=0,
        frozen_by="compactor",
        frozen_at_utc="2026-07-27T13:00:00+00:00",
    )
    return root


def test_local_compaction_writes_valid_multisubject_successor_and_receipt(
    tmp_path: Path,
) -> None:
    base = _base_publication(tmp_path)
    delta_root = _frozen_delta(tmp_path, base)
    base_hashes_before = dict(base.receipt["logical_hashes"])
    destination_root = tmp_path / "compactions"

    result = compact_frozen_refined_detection_delta_generation(
        delta_root=delta_root,
        delta_lineage_id=DELTA_LINEAGE_ID,
        generation_ordinal=0,
        base_manifest=base.manifest,
        base_arrays=base.arrays,
        destination=destination_root / "successor.zarr",
        run_id="refined_successor",
        snapshot_id=SUCCESSOR_SNAPSHOT_ID,
        created_by="test_compactor",
        safe_root=destination_root,
    )

    assert result.publication.dimensions.n_instances == 4
    np.testing.assert_array_equal(
        result.publication.arrays["instances/frame_indices"][:],
        [0, 0, 1, 2],
    )
    np.testing.assert_array_equal(
        result.publication.arrays["instances/frame_row_offsets"][:],
        [0, 2, 3, 4, 4],
    )
    assert (
        result.publication.manifest["payload"]["snapshot_lineage"]["parent_snapshot"][
            "run_manifest_digest"
        ]
        == base.manifest["payload_digest"]
    )
    assert (
        result.publication.manifest["payload"]["publication"]["stage_selector_eligible"]
        is False
    )
    assert result.receipt["payload"]["production_state_changes"] == []
    assert result.receipt["payload"]["local_store"] is True
    assert validate_refined_detection_compaction_receipt(result.receipt) == ()
    persisted = json.loads(
        (
            result.publication.output_path / REFINED_DETECTION_COMPACTION_RECEIPT_NAME
        ).read_text(encoding="utf-8")
    )
    assert persisted == result.receipt
    assert base_hashes_before == refined_detection_logical_hashes(base.arrays)
    phase_seconds = result.receipt["payload"]["phase_seconds"]
    assert {
        "read_and_validate_base",
        "read_and_verify_frozen_delta_prefix",
        "resolve_sort_and_rebuild_offsets",
        "publish_validate_immutable_snapshot",
        "snapshot_publication",
        "per_array_write",
        "total_before_receipt",
    } == set(phase_seconds)
    assert all(
        float(value) >= 0
        for name, value in phase_seconds.items()
        if name not in {"snapshot_publication", "per_array_write"}
    )
    assert not (
        base.output_path / "refined_detect_runs" / "refined_base" / "instances"
    ).is_symlink()


def test_compaction_preserves_coordinate_catalog_manifest_version(
    tmp_path: Path,
) -> None:
    base = _base_publication(tmp_path, coordinate_catalog=True)
    destination_root = tmp_path / "compactions"
    result = compact_frozen_refined_detection_delta_generation(
        delta_root=_frozen_delta(tmp_path, base),
        delta_lineage_id=DELTA_LINEAGE_ID,
        generation_ordinal=0,
        base_manifest=base.manifest,
        base_arrays=base.arrays,
        destination=destination_root / "successor.zarr",
        run_id="refined_successor",
        snapshot_id=SUCCESSOR_SNAPSHOT_ID,
        created_by="test_compactor",
        safe_root=destination_root,
    )

    assert result.publication.manifest["schema_version"] == (
        REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert "coordinate_contract" in result.publication.manifest["payload"]
    assert result.receipt["payload"]["output"]["run_manifest_schema_version"] == (
        REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )


def test_compaction_receipt_tampering_and_unsafe_destinations_fail_closed(
    tmp_path: Path,
) -> None:
    base = _base_publication(tmp_path)
    result = compact_frozen_refined_detection_delta_generation(
        delta_root=_frozen_delta(tmp_path, base),
        delta_lineage_id=DELTA_LINEAGE_ID,
        generation_ordinal=0,
        base_manifest=base.manifest,
        base_arrays=base.arrays,
        destination=tmp_path / "compactions/successor.zarr",
        run_id="refined_successor",
        snapshot_id=SUCCESSOR_SNAPSHOT_ID,
        created_by="test_compactor",
        safe_root=tmp_path / "compactions",
    )
    tampered = copy.deepcopy(result.receipt)
    tampered["payload"]["selector_eligible"] = True
    assert "compaction receipt payload digest mismatch" in (
        validate_refined_detection_compaction_receipt(tampered)
    )
    assert "compaction receipt must remain selector-ineligible" in (
        validate_refined_detection_compaction_receipt(tampered)
    )

    unsafe_root = Path("/var/tmp/palette-output")
    with pytest.raises(ValueError, match="Snapshot roots must be below"):
        require_safe_refined_detection_snapshot_destination(
            unsafe_root / "candidate.zarr",
            safe_root=unsafe_root,
        )


def test_compaction_rejects_schema_valid_base_payload_with_wrong_content(
    tmp_path: Path,
) -> None:
    base = _base_publication(tmp_path)
    delta_root = _frozen_delta(tmp_path, base)
    wrong_base = {
        path: np.asarray(array[:]).copy() for path, array in base.arrays.items()
    }
    wrong_base["instances/scores"][0] = np.float32(0.1)
    wrong_base["source_detections/scores"][0] = np.float32(0.1)
    destination_root = tmp_path / "compactions"

    with pytest.raises(
        RefinedDetectionCompactionError,
        match="does not bind the supplied immutable base",
    ):
        compact_frozen_refined_detection_delta_generation(
            delta_root=delta_root,
            delta_lineage_id=DELTA_LINEAGE_ID,
            generation_ordinal=0,
            base_manifest=base.manifest,
            base_arrays=wrong_base,
            destination=destination_root / "wrong_base.zarr",
            run_id="refined_wrong_base",
            snapshot_id=SUCCESSOR_SNAPSHOT_ID,
            created_by="test_compactor",
            safe_root=destination_root,
        )
    assert not (destination_root / "wrong_base.zarr").exists()
