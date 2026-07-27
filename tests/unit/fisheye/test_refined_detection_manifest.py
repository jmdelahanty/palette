from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from fisheye.shared.instance_keys import mint_manual_curation_instance_keys
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE,
    REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE,
    REFINED_DETECTION_RUN_MANIFEST_PERSISTED_PATH,
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceIdentity,
    build_refined_detection_authority_provenance,
    build_refined_detection_run_manifest,
    canonical_json_sha256,
    refined_detection_selection_contract_manifest,
    validate_refined_detection_authority_provenance,
    validate_refined_detection_run_manifest,
    validate_refined_detection_snapshot_identity,
)
from fisheye.shared.zarr.refined_detection_schema import (
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionClipBinding,
    RefinedDetectionClippedBinding,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_storage import (
    REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    plan_refined_detection_storage,
)


LINEAGE_ID = "11111111-1111-4111-8111-111111111111"
ROOT_SNAPSHOT_ID = "22222222-2222-4222-8222-222222222222"
NEXT_SNAPSHOT_ID = "33333333-3333-4333-8333-333333333333"


def _dimensions(n_instances: int) -> RefinedDetectionDimensions:
    return RefinedDetectionDimensions(
        n_frames=4,
        n_instances=n_instances,
        n_source_detections=1,
        source_width=640,
        source_height=480,
    )


def _lineage(
    *,
    snapshot_id: str,
    next_id: int,
    parent_run_id: str | None = None,
    parent_digest: str | None = None,
) -> RefinedDetectionSnapshotLineage:
    return RefinedDetectionSnapshotLineage(
        lineage_id=LINEAGE_ID,
        snapshot_id=snapshot_id,
        recording_identity="sleepyfish_cam2010095",
        next_refined_row_id=next_id,
        parent_run_id=parent_run_id,
        parent_manifest_digest=parent_digest,
    )


def _build_manifest(
    *,
    run_id: str,
    dimensions: RefinedDetectionDimensions,
    lineage: RefinedDetectionSnapshotLineage,
) -> dict[str, object]:
    return build_refined_detection_run_manifest(
        run_id=run_id,
        dimensions=dimensions,
        storage_plan=plan_refined_detection_storage(
            dimensions,
            profile=REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
        ),
        lineage=lineage,
        source=RefinedDetectionSourceIdentity(
            run_id="detect_1",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        instance_reason_codes={0: "none", 1: "manual_addition"},
        source_reason_codes={0: "none", 1: "filtered_low_score"},
        metadata_declarations_digest="c" * 64,
        selector_eligible=True,
    )


def _manual_arrays(
    row_ids: list[int],
    *,
    bboxes: np.ndarray | None = None,
    keys: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rows = len(row_ids)
    frames = np.arange(rows, dtype=np.int32)
    resolved_bboxes = (
        np.asarray(bboxes, dtype=np.float32).reshape(rows, 4)
        if bboxes is not None
        else np.asarray(
            [[0.5, 0.5, 0.2, 0.2] for _ in row_ids],
            dtype=np.float32,
        )
    )
    classes = np.ones(rows, dtype=np.int32)
    row_id_array = np.asarray(row_ids, dtype=np.int64)
    resolved_keys = (
        np.asarray(keys, dtype=np.uint64)
        if keys is not None
        else mint_manual_curation_instance_keys(
            recording_identity="sleepyfish_cam2010095",
            refined_row_ids=row_id_array,
            frame_indices=frames,
            bbox_norm_coords=resolved_bboxes,
            class_ids=classes,
        )
    )
    return {
        "instances/refined_row_ids": row_id_array,
        "instances/instance_key": resolved_keys,
        "instances/source_kind_codes": np.full(
            rows,
            SOURCE_KIND_CODE_MAP["manual"],
            dtype=np.uint8,
        ),
        "instances/frame_indices": frames,
        "instances/bbox_norm_coords": resolved_bboxes,
        "instances/class_ids": classes,
    }


def test_run_manifest_freezes_path_digest_publication_and_separate_reasons() -> None:
    dimensions = _dimensions(1)
    manifest = _build_manifest(
        run_id="refined_1",
        dimensions=dimensions,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
    )

    assert REFINED_DETECTION_RUN_MANIFEST_PERSISTED_PATH == (
        "refined_detect_runs/<run>/zarr.json.attributes.run_manifest"
    )
    assert json.loads(json.dumps(manifest)) == manifest
    assert validate_refined_detection_run_manifest(manifest) == ()
    payload = manifest["payload"]
    assert payload["publication"]["completion_status"] == "complete"
    assert payload["publication"]["stage_selector_eligible"] is True
    assert payload["logical_schema"]["schema_id"] == (
        "palette.stage.refined_detection"
    )
    assert payload["reason_registries"]["instances"]["codes"] == {
        "0": "none",
        "1": "manual_addition",
    }
    assert payload["reason_registries"]["source_detections"]["codes"] == {
        "0": "none",
        "1": "filtered_low_score",
    }
    assert (
        payload["reason_registries"]["instances"]["digest"]
        != payload["reason_registries"]["source_detections"]["digest"]
    )

    tampered = copy.deepcopy(manifest)
    tampered["payload"]["run_id"] = "tampered"
    assert "run manifest payload_digest mismatch" in (
        validate_refined_detection_run_manifest(tampered)
    )

    semantically_tampered = copy.deepcopy(manifest)
    semantically_tampered["payload"]["reason_registries"]["instances"][
        "codes"
    ]["1"] = "changed"
    semantically_tampered["payload_digest"] = canonical_json_sha256(
        semantically_tampered["payload"]
    )
    assert "instances reason registry digest mismatch" in (
        validate_refined_detection_run_manifest(semantically_tampered)
    )


def test_manifest_rejects_ambiguous_reason_registry_and_zero_frame_dimension() -> None:
    dimensions = _dimensions(1)
    with pytest.raises(ValueError, match="code 0"):
        build_refined_detection_run_manifest(
            run_id="refined_1",
            dimensions=dimensions,
            storage_plan=plan_refined_detection_storage(dimensions),
            lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
            source=RefinedDetectionSourceIdentity(
                run_id="detect_1",
                run_manifest_digest="a" * 64,
                logical_content_digest="b" * 64,
            ),
            instance_reason_codes={1: "manual_addition"},
            source_reason_codes={0: "none"},
            metadata_declarations_digest="c" * 64,
            selector_eligible=True,
        )


def test_clipped_run_manifest_binds_one_camera_and_complete_media_timeline() -> None:
    dimensions = RefinedDetectionDimensions(
        n_frames=4,
        n_instances=1,
        n_source_detections=1,
        source_width=640,
        source_height=480,
        lineage_profile=RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT,
    )
    clipped = RefinedDetectionClippedBinding(
        collection_id="collection_1",
        collection_manifest_digest="1" * 64,
        camera_serial="2010095",
        video_identity="sleepyfish_cam2010095",
        video_manifest_digest="2" * 64,
        recording_frame_index_digest="3" * 64,
        clips=(
            RefinedDetectionClipBinding(
                clip_index=0,
                clip_id="clip_0",
                media_identity="clip_0.mp4",
                media_digest="4" * 64,
                parent_frame_start=0,
                parent_frame_stop=4,
                frame_map_digest="5" * 64,
                source_refined_run_id="refined_clip_0",
                source_refined_manifest_digest="6" * 64,
            ),
        ),
    )
    manifest = build_refined_detection_run_manifest(
        run_id="refined_clipped_1",
        dimensions=dimensions,
        storage_plan=plan_refined_detection_storage(dimensions),
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
        source=RefinedDetectionSourceIdentity(
            run_id="detect_1",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        instance_reason_codes={0: "none"},
        source_reason_codes={0: "none"},
        metadata_declarations_digest="c" * 64,
        selector_eligible=True,
        clipped_binding=clipped,
    )

    assert validate_refined_detection_run_manifest(manifest) == ()
    binding = manifest["payload"]["logical_schema"]["clipped_binding"]
    assert binding["camera_cardinality"] == 1
    assert binding["clip_ordinal_scope"] == (
        "snapshot_global_within_single_camera"
    )
    assert binding["empty_frame_media_resolution"] == (
        "complete_frame_map_independent_of_rows"
    )


def test_root_snapshot_validates_manual_key_allocator() -> None:
    dimensions = _dimensions(1)
    manifest = _build_manifest(
        run_id="refined_1",
        dimensions=dimensions,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
    )
    arrays = _manual_arrays([0])

    assert validate_refined_detection_snapshot_identity(
        manifest=manifest,
        arrays=arrays,
    ) == ()

    arrays["instances/instance_key"][0] += np.uint64(1)
    assert "manual instance_key values do not match the frozen allocator" in (
        validate_refined_detection_snapshot_identity(
            manifest=manifest,
            arrays=arrays,
        )
    )


def test_successor_enforces_parent_digest_nonreuse_and_surviving_key() -> None:
    parent_dimensions = _dimensions(1)
    parent_manifest = _build_manifest(
        run_id="refined_1",
        dimensions=parent_dimensions,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=2),
    )
    parent_arrays = _manual_arrays([0])
    parent_digest = str(parent_manifest["payload_digest"])

    current_dimensions = _dimensions(2)
    current_lineage = _lineage(
        snapshot_id=NEXT_SNAPSHOT_ID,
        next_id=3,
        parent_run_id="refined_1",
        parent_digest=parent_digest,
    )
    current_manifest = _build_manifest(
        run_id="refined_2",
        dimensions=current_dimensions,
        lineage=current_lineage,
    )
    edited_bbox = np.asarray(
        [[0.55, 0.50, 0.20, 0.20], [0.4, 0.4, 0.1, 0.1]],
        dtype=np.float32,
    )
    new_key = mint_manual_curation_instance_keys(
        recording_identity="sleepyfish_cam2010095",
        refined_row_ids=np.asarray([2], dtype=np.int64),
        frame_indices=np.asarray([1], dtype=np.int32),
        bbox_norm_coords=edited_bbox[1:2],
        class_ids=np.asarray([1], dtype=np.int32),
    )
    current_arrays = _manual_arrays(
        [0, 2],
        bboxes=edited_bbox,
        keys=np.asarray(
            [parent_arrays["instances/instance_key"][0], new_key[0]],
            dtype=np.uint64,
        ),
    )

    assert validate_refined_detection_snapshot_identity(
        manifest=current_manifest,
        arrays=current_arrays,
        parent_manifest=parent_manifest,
        parent_arrays=parent_arrays,
    ) == ()

    reused = _manual_arrays([0, 1], bboxes=edited_bbox)
    errors = validate_refined_detection_snapshot_identity(
        manifest=current_manifest,
        arrays=reused,
        parent_manifest=parent_manifest,
        parent_arrays=parent_arrays,
    )
    assert "retired refined_row_id 1 was reused by successor" in errors

    changed_key = copy.deepcopy(current_arrays)
    changed_key["instances/instance_key"][0] += np.uint64(1)
    errors = validate_refined_detection_snapshot_identity(
        manifest=current_manifest,
        arrays=changed_key,
        parent_manifest=parent_manifest,
        parent_arrays=parent_arrays,
    )
    assert "surviving refined_row_id 0 changed instance_key" in errors


def test_authority_envelope_and_selection_are_typed_and_fail_closed() -> None:
    authority = build_refined_detection_authority_provenance(
        run_id="refined_1",
        run_manifest_digest="a" * 64,
        approved_by="reviewer",
        approved_at_utc="2026-07-27T12:00:00+00:00",
        review_method="manual",
        intended_use="training",
        git_sha="abc123",
    )
    selection = refined_detection_selection_contract_manifest()

    assert REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE == "authoritative_run"
    assert REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE == (
        "authoritative_run_provenance"
    )
    assert authority["payload"]["review_state"] == "approved"
    assert authority["payload"]["intended_use"] == "training"
    assert validate_refined_detection_authority_provenance(authority) == ()
    assert selection["order"] == [
        "explicit_refined_v1",
        "approved_authoritative_refined_v1",
        "explicitly_permitted_canonical_raw",
    ]
    assert selection["request"]["default_raw_fallback_policy"] == "forbid"
    assert selection["explicit_refined_v1"]["failure"] == (
        "terminal_error_never_raw_fallback"
    )
    assert selection["approved_authoritative_refined_v1"]["invalid_pointer"] == (
        "terminal_error_never_raw_fallback"
    )

    tampered = copy.deepcopy(authority)
    tampered["payload"]["review_state"] = "needs_review"
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert "authority review_state must be approved" in (
        validate_refined_detection_authority_provenance(tampered)
    )

    with pytest.raises(ValueError, match="intended_use"):
        build_refined_detection_authority_provenance(
            run_id="refined_1",
            run_manifest_digest="a" * 64,
            approved_by="reviewer",
            approved_at_utc="2026-07-27T12:00:00+00:00",
            review_method="manual",
            intended_use="anything",
        )
