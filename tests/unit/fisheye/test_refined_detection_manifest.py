from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from fisheye.shared.zarr.array_factory import array_metadata_declaration_from_plan
from fisheye.shared.instance_keys import mint_manual_curation_instance_keys
from fisheye.shared.zarr.detection_schema import (
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE,
    REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE,
    REFINED_DETECTION_RUN_MANIFEST_PERSISTED_PATH,
    RefinedDetectionBoundClipEvidence,
    RefinedDetectionClipSourceIdentity,
    RefinedDetectionSnapshotLineage,
    RefinedDetectionSourceCollectionIdentity,
    RefinedDetectionSourceIdentity,
    build_refined_detection_activation_candidate_manifest,
    build_refined_detection_authority_provenance,
    build_coordinate_refined_detection_run_manifest,
    build_refined_detection_run_manifest,
    canonical_json_sha256,
    normalize_refined_detection_metadata_declarations,
    refined_detection_metadata_declarations_digest,
    refined_detection_selection_contract_manifest,
    validate_refined_detection_authority_provenance,
    validate_refined_detection_clipped_source_evidence,
    validate_refined_detection_publication,
    validate_refined_detection_reason_code_coverage,
    validate_refined_detection_run_manifest,
    validate_refined_detection_snapshot_identity,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_DECISION_CODE_MAP,
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
    selector_eligible: bool = True,
) -> dict[str, object]:
    storage_plan = plan_refined_detection_storage(
        dimensions,
        profile=REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    )
    direct, consolidated = _metadata_declarations(dimensions, storage_plan)
    return build_refined_detection_run_manifest(
        run_id=run_id,
        dimensions=dimensions,
        storage_plan=storage_plan,
        lineage=lineage,
        source=RefinedDetectionSourceIdentity(
            run_id="detect_1",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        instance_reason_codes={0: "none", 1: "manual_addition"},
        source_reason_codes={0: "none", 1: "filtered_low_score"},
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=selector_eligible,
    )


def _metadata_declarations(
    dimensions: RefinedDetectionDimensions,
    storage_plan=None,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    plan = storage_plan or plan_refined_detection_storage(dimensions)
    declarations: dict[str, dict[str, object]] = {
        "": {"zarr_format": 3, "node_type": "group", "attributes": {}},
        "instances": {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
        },
        "source_detections": {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
        },
    }
    binding_by_path = {
        binding.path: binding
        for binding in REFINED_DETECTION_SCHEMA_V1.bindings_for(dimensions)
    }
    for entry in plan.entries:
        physical = entry.plan
        binding = binding_by_path[entry.rule.path]
        contract = REFINED_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        declarations[entry.rule.path] = {
            "zarr_format": 3,
            "node_type": "array",
            **array_metadata_declaration_from_plan(
                contract=contract,
                plan=physical,
                fill_value=False if physical.logical_dtype == "bool" else 0,
                attributes={"ignored_by_declaration_digest": True},
            ),
        }
    return declarations, copy.deepcopy(declarations)


def _empty_arrays(
    dimensions: RefinedDetectionDimensions,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for binding in REFINED_DETECTION_SCHEMA_V1.bindings_for(dimensions):
        contract = REFINED_DETECTION_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        shape = tuple(
            value if isinstance(value, int) else dimensions.contract_dimensions[value]
            for value in contract.shape_template
        )
        arrays[binding.path] = np.zeros(shape, dtype=contract.dtype.numpy_dtype)
    return arrays


def _complete_manual_arrays_same_frame(
    dimensions: RefinedDetectionDimensions,
) -> dict[str, np.ndarray]:
    arrays = _empty_arrays(dimensions)
    rows = dimensions.n_instances
    assert rows == 2
    frames = np.asarray([1, 1], dtype=np.int32)
    row_ids = np.asarray([0, 1], dtype=np.int64)
    bboxes = np.asarray(
        [[0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]],
        dtype=np.float32,
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        bboxes,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    classes = np.asarray([1, 1], dtype=np.int32)
    arrays.update(
        {
            "instances/frame_indices": frames,
            "instances/source_acquisition_frame_index": frames.astype(np.int64),
            "instances/instance_key": mint_manual_curation_instance_keys(
                recording_identity="sleepyfish_cam2010095",
                refined_row_ids=row_ids,
                frame_indices=frames,
                bbox_norm_coords=bboxes,
                class_ids=classes,
            ),
            "instances/refined_row_ids": row_ids,
            "instances/bbox_norm_coords": bboxes,
            "instances/bbox_img_xyxy": bbox_img,
            "instances/centers_img_xy": centers,
            "instances/scores": np.zeros(rows, dtype=np.float32),
            "instances/score_valid": np.zeros(rows, dtype=np.bool_),
            "instances/class_ids": classes,
            "instances/source_kind_codes": np.full(
                rows,
                SOURCE_KIND_CODE_MAP["manual"],
                dtype=np.uint8,
            ),
            "instances/manual_edit_flags": np.ones(rows, dtype=np.bool_),
            "instances/source_detect_row_index": np.full(
                rows,
                -1,
                dtype=np.int64,
            ),
            "instances/reason_codes": np.zeros(rows, dtype=np.uint16),
            "instances/frame_row_offsets": np.asarray(
                [0, 0, 2, 2, 2],
                dtype=np.int64,
            ),
        }
    )
    return arrays


def _one_raw_clip_arrays(
    dimensions: RefinedDetectionDimensions,
) -> dict[str, np.ndarray]:
    arrays = _empty_arrays(dimensions)
    bbox = np.asarray([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    bbox_img, centers = derive_canonical_detection_geometry(
        bbox,
        source_width=dimensions.source_width,
        source_height=dimensions.source_height,
    )
    shared = {
        "frame_indices": np.asarray([0], dtype=np.int32),
        "source_acquisition_frame_index": np.asarray([0], dtype=np.int64),
        "instance_key": np.asarray([123], dtype=np.uint64),
        "bbox_norm_coords": bbox,
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers,
        "scores": np.asarray([0.8], dtype=np.float32),
        "class_ids": np.asarray([1], dtype=np.int32),
    }
    arrays.update(
        {
            **{f"instances/{name}": value.copy() for name, value in shared.items()},
            "instances/refined_row_ids": np.asarray([7], dtype=np.int64),
            "instances/score_valid": np.asarray([True], dtype=np.bool_),
            "instances/source_kind_codes": np.asarray(
                [SOURCE_KIND_CODE_MAP["raw_detect"]],
                dtype=np.uint8,
            ),
            "instances/manual_edit_flags": np.asarray([False], dtype=np.bool_),
            "instances/source_detect_row_index": np.asarray([0], dtype=np.int64),
            "instances/reason_codes": np.asarray([0], dtype=np.uint16),
            "instances/frame_row_offsets": np.asarray([0, 1, 1], dtype=np.int64),
            **{
                f"source_detections/{name}": value.copy()
                for name, value in shared.items()
            },
            "source_detections/source_detect_row_index": np.asarray(
                [0],
                dtype=np.int64,
            ),
            "source_detections/decision_codes": np.asarray(
                [SOURCE_DECISION_CODE_MAP["accepted"]],
                dtype=np.uint8,
            ),
            "source_detections/resolved_refined_row_id": np.asarray(
                [7],
                dtype=np.int64,
            ),
            "source_detections/reason_codes": np.asarray([0], dtype=np.uint16),
            "source_detections/frame_row_offsets": np.asarray(
                [0, 1, 1],
                dtype=np.int64,
            ),
        }
    )
    return arrays


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
    storage_plan = plan_refined_detection_storage(
        dimensions,
        profile=REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    )
    direct, consolidated = _metadata_declarations(dimensions, storage_plan)
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
    assert manifest["schema_version"] == 1
    assert "coordinate_contract" not in payload
    assert payload["publication"]["completion_status"] == "complete"
    assert payload["publication"]["stage_selector_eligible"] is True
    assert payload["publication"]["metadata_declarations_digest"] == (
        refined_detection_metadata_declarations_digest(
            direct,
            consolidated_metadata_by_path=consolidated,
            dimensions=dimensions,
        )
    )
    assert payload["logical_schema"]["schema_id"] == ("palette.stage.refined_detection")
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
    semantically_tampered["payload"]["reason_registries"]["instances"]["codes"]["1"] = (
        "changed"
    )
    semantically_tampered["payload_digest"] = canonical_json_sha256(
        semantically_tampered["payload"]
    )
    assert "instances reason registry is not in canonical persisted form" in (
        validate_refined_detection_run_manifest(semantically_tampered)
    )


def test_opt_in_refined_manifest_persists_exact_coordinate_catalog() -> None:
    dimensions = _dimensions(1)
    storage_plan = plan_refined_detection_storage(
        dimensions,
        profile=REFINED_DETECTION_ACCESS_AWARE_CANDIDATE_V1,
    )
    direct, consolidated = _metadata_declarations(dimensions, storage_plan)
    kwargs = {
        "run_id": "refined_coordinate_1",
        "dimensions": dimensions,
        "storage_plan": storage_plan,
        "lineage": _lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
        "source": RefinedDetectionSourceIdentity(
            run_id="detect_1",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        "instance_reason_codes": {0: "none", 1: "manual_addition"},
        "source_reason_codes": {0: "none", 1: "filtered_low_score"},
        "direct_metadata_declarations": direct,
        "consolidated_metadata_declarations": consolidated,
        "selector_eligible": False,
    }
    legacy = build_refined_detection_run_manifest(**kwargs)
    manifest = build_coordinate_refined_detection_run_manifest(**kwargs)

    assert legacy["schema_version"] == 1
    assert "coordinate_contract" not in legacy["payload"]
    assert manifest["schema_version"] == (
        REFINED_DETECTION_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert manifest["payload"]["coordinate_contract"]["document"] == (
        REFINED_DETECTION_SCHEMA_V1.coordinate_contract_manifest()
    )
    assert validate_refined_detection_run_manifest(manifest) == ()

    tampered = copy.deepcopy(manifest)
    catalog = tampered["payload"]["coordinate_contract"]
    catalog["document"]["bindings"][0]["surface_id"] = "wrong_surface"
    catalog["digest"] = canonical_json_sha256(catalog["document"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert "coordinate catalog differs from the frozen stage catalog" in (
        validate_refined_detection_run_manifest(tampered)
    )


def test_recomputed_digest_cannot_hide_nested_contract_tampering() -> None:
    dimensions = _dimensions(1)
    manifest = _build_manifest(
        run_id="refined_1",
        dimensions=dimensions,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
    )

    variants: list[tuple[dict[str, object], str]] = []
    logical = copy.deepcopy(manifest)
    logical["payload"]["logical_schema"]["array_contracts"]["contracts"][0]["dtype"][
        "dtype_id"
    ] = "float64"
    variants.append((logical, "logical_schema"))

    codec = copy.deepcopy(manifest)
    codec["payload"]["storage_plan"]["codec_profile"]["codec_chain"][1][
        "configuration"
    ]["level"] = 9
    variants.append((codec, "storage_plan"))

    chunks = copy.deepcopy(manifest)
    chunks["payload"]["storage_plan"]["arrays"][0]["plan"]["chunk_shape"][0] = 999
    variants.append((chunks, "storage_plan"))

    ownership = copy.deepcopy(manifest)
    ownership["payload"]["storage_plan"]["arrays"][0]["plan"]["write_ownership"] = (
        "partial_chunk_writes"
    )
    variants.append((ownership, "storage_plan"))

    estimates = copy.deepcopy(manifest)
    estimates["payload"]["storage_plan"]["object_estimate"]["stage_objects"] += 1
    variants.append((estimates, "storage_plan"))

    unexpected = copy.deepcopy(manifest)
    unexpected["payload"]["publication"]["unexpected"] = True
    variants.append((unexpected, "publication"))

    for tampered, expected_error in variants:
        tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
        assert any(
            expected_error in error
            for error in validate_refined_detection_run_manifest(tampered)
        )


def test_metadata_declaration_normalizer_is_exact_and_checks_consolidation() -> None:
    dimensions = _dimensions(1)
    direct, consolidated = _metadata_declarations(dimensions)
    for path in ("", "instances", "source_detections"):
        direct[path]["consolidated_metadata"] = None
        consolidated[path]["consolidated_metadata"] = {
            "kind": "inline",
            "must_understand": False,
            "metadata": {},
        }
    normalized = normalize_refined_detection_metadata_declarations(
        direct,
        consolidated_metadata_by_path=consolidated,
        dimensions=dimensions,
    )

    assert set(normalized["declarations"]) == set(direct)
    assert all(
        "attributes" not in declaration
        for declaration in normalized["declarations"].values()
    )
    direct_attributes_changed = copy.deepcopy(direct)
    consolidated_attributes_changed = copy.deepcopy(consolidated)
    direct_attributes_changed["instances/frame_indices"]["attributes"]["another"] = 1
    consolidated_attributes_changed["instances/frame_indices"]["attributes"][
        "another"
    ] = 1
    assert refined_detection_metadata_declarations_digest(
        direct_attributes_changed,
        consolidated_metadata_by_path=consolidated_attributes_changed,
        dimensions=dimensions,
    ) == refined_detection_metadata_declarations_digest(
        direct,
        consolidated_metadata_by_path=consolidated,
        dimensions=dimensions,
    )

    mismatched = copy.deepcopy(direct)
    mismatched["instances/frame_indices"]["shape"] = [999]
    with pytest.raises(ValueError, match="Direct and consolidated metadata differ"):
        normalize_refined_detection_metadata_declarations(
            mismatched,
            consolidated_metadata_by_path=consolidated,
            dimensions=dimensions,
        )

    missing = copy.deepcopy(direct)
    del missing["instances/frame_indices"]
    with pytest.raises(ValueError, match="paths must be exact"):
        normalize_refined_detection_metadata_declarations(
            missing,
            consolidated_metadata_by_path=consolidated,
            dimensions=dimensions,
        )

    nonempty_group_envelope = copy.deepcopy(consolidated)
    nonempty_group_envelope["instances"]["consolidated_metadata"]["metadata"] = {
        "unexpected": {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {},
        }
    }
    with pytest.raises(ValueError, match="exact empty inline group envelope"):
        normalize_refined_detection_metadata_declarations(
            direct,
            consolidated_metadata_by_path=nonempty_group_envelope,
            dimensions=dimensions,
        )

    array_level_envelope = copy.deepcopy(consolidated)
    array_level_envelope["instances/frame_indices"]["consolidated_metadata"] = None
    with pytest.raises(ValueError, match="Only Zarr groups"):
        normalize_refined_detection_metadata_declarations(
            direct,
            consolidated_metadata_by_path=array_level_envelope,
            dimensions=dimensions,
        )


def test_metadata_normalizer_allows_only_exact_final_eligibility_commit_lag() -> None:
    dimensions = _dimensions(1)
    direct, consolidated = _metadata_declarations(dimensions)
    manifest = _build_manifest(
        run_id="refined_activated",
        dimensions=dimensions,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
        selector_eligible=True,
    )
    direct[""]["attributes"] = {
        "run_manifest": manifest,
        "stage_selector_eligible": True,
    }
    consolidated[""]["attributes"] = {
        "run_manifest": copy.deepcopy(manifest),
        "stage_selector_eligible": False,
    }

    normalized = normalize_refined_detection_metadata_declarations(
        direct,
        consolidated_metadata_by_path=consolidated,
        dimensions=dimensions,
    )
    assert "attributes" not in normalized["declarations"][""]

    wrong_intent = copy.deepcopy(direct)
    wrong_intent_manifest = copy.deepcopy(manifest)
    wrong_intent_manifest["payload"]["publication"][
        "stage_selector_eligible"
    ] = False
    wrong_intent_manifest["payload_digest"] = canonical_json_sha256(
        wrong_intent_manifest["payload"]
    )
    wrong_intent[""]["attributes"]["run_manifest"] = wrong_intent_manifest
    wrong_intent_consolidated = copy.deepcopy(consolidated)
    wrong_intent_consolidated[""]["attributes"][
        "run_manifest"
    ] = copy.deepcopy(wrong_intent_manifest)
    with pytest.raises(ValueError, match="Direct and consolidated metadata differ"):
        normalize_refined_detection_metadata_declarations(
            wrong_intent,
            consolidated_metadata_by_path=wrong_intent_consolidated,
            dimensions=dimensions,
        )

    unrelated_drift = copy.deepcopy(direct)
    unrelated_drift[""]["attributes"]["unexpected"] = "changed"
    with pytest.raises(ValueError, match="Direct and consolidated metadata differ"):
        normalize_refined_detection_metadata_declarations(
            unrelated_drift,
            consolidated_metadata_by_path=consolidated,
            dimensions=dimensions,
        )


def test_manifest_rejects_ambiguous_reason_registry_and_zero_frame_dimension() -> None:
    dimensions = _dimensions(1)
    storage_plan = plan_refined_detection_storage(dimensions)
    direct, consolidated = _metadata_declarations(dimensions, storage_plan)
    with pytest.raises(ValueError, match="code 0"):
        build_refined_detection_run_manifest(
            run_id="refined_1",
            dimensions=dimensions,
            storage_plan=storage_plan,
            lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
            source=RefinedDetectionSourceIdentity(
                run_id="detect_1",
                run_manifest_digest="a" * 64,
                logical_content_digest="b" * 64,
            ),
            instance_reason_codes={1: "manual_addition"},
            source_reason_codes={0: "none"},
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            selector_eligible=True,
        )


@pytest.mark.parametrize(
    ("bad_code", "bad_label"),
    (("01", "manual_addition"), ("1", "ManualAddition"), ("65536", "too_large")),
)
def test_parsed_reason_registries_reject_noncanonical_codes_and_labels(
    bad_code: str,
    bad_label: str,
) -> None:
    dimensions = _dimensions(1)
    manifest = _build_manifest(
        run_id="refined_1",
        dimensions=dimensions,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
    )
    tampered = copy.deepcopy(manifest)
    registry = tampered["payload"]["reason_registries"]["instances"]
    registry["codes"] = {"0": "none", bad_code: bad_label}
    registry_payload = {
        "schema_id": registry["schema_id"],
        "schema_version": registry["schema_version"],
        "registry_id": registry["registry_id"],
        "codes": registry["codes"],
    }
    registry["digest"] = canonical_json_sha256(registry_payload)
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    assert any(
        "instances reason registry is invalid" in error
        for error in validate_refined_detection_run_manifest(tampered)
    )


def test_reason_registry_coverage_rejects_unregistered_persisted_codes() -> None:
    dimensions = _dimensions(1)
    manifest = _build_manifest(
        run_id="refined_1",
        dimensions=dimensions,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
    )
    arrays = {
        "instances/reason_codes": np.asarray([0, 2], dtype=np.uint16),
        "source_detections/reason_codes": np.asarray([0, 1], dtype=np.uint16),
    }

    assert validate_refined_detection_reason_code_coverage(
        manifest,
        arrays,
    ) == ("reason-code array 'instances/reason_codes' contains unregistered codes [2]",)


def test_publication_gate_recomputes_metadata_digest_and_validates_arrays() -> None:
    dimensions = RefinedDetectionDimensions(
        n_frames=4,
        n_instances=0,
        n_source_detections=0,
        source_width=640,
        source_height=480,
    )
    storage_plan = plan_refined_detection_storage(dimensions)
    direct, consolidated = _metadata_declarations(dimensions, storage_plan)
    manifest = build_refined_detection_run_manifest(
        run_id="refined_empty_1",
        dimensions=dimensions,
        storage_plan=storage_plan,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=0),
        source=RefinedDetectionSourceIdentity(
            run_id="detect_1",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        instance_reason_codes={0: "none"},
        source_reason_codes={0: "none"},
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )
    arrays = _empty_arrays(dimensions)

    assert (
        validate_refined_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
        )
        == ()
    )

    physically_changed = copy.deepcopy(direct)
    physically_changed["instances/frame_indices"]["shape"] = [1]
    consolidated_changed = copy.deepcopy(consolidated)
    consolidated_changed["instances/frame_indices"]["shape"] = [1]
    assert "metadata_declarations_digest does not match declarations" in (
        validate_refined_detection_publication(
            manifest,
            direct_metadata_declarations=physically_changed,
            consolidated_metadata_declarations=consolidated_changed,
            arrays=arrays,
        )
    )

    replanned_metadata = copy.deepcopy(direct)
    replanned_metadata["instances/frame_indices"]["codecs"][1]["configuration"][
        "level"
    ] = 7
    replanned_consolidated = copy.deepcopy(replanned_metadata)
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["payload"]["publication"]["metadata_declarations_digest"] = (
        refined_detection_metadata_declarations_digest(
            replanned_metadata,
            consolidated_metadata_by_path=replanned_consolidated,
            dimensions=dimensions,
        )
    )
    tampered_manifest["payload_digest"] = canonical_json_sha256(
        tampered_manifest["payload"]
    )
    assert any(
        "physical metadata at instances/frame_indices" in error
        for error in validate_refined_detection_publication(
            tampered_manifest,
            direct_metadata_declarations=replanned_metadata,
            consolidated_metadata_declarations=replanned_consolidated,
            arrays=arrays,
        )
    )


def test_publication_gate_enforces_manual_keys_with_multiple_subjects_per_frame() -> (
    None
):
    dimensions = RefinedDetectionDimensions(
        n_frames=4,
        n_instances=2,
        n_source_detections=0,
        source_width=640,
        source_height=480,
    )
    storage_plan = plan_refined_detection_storage(dimensions)
    direct, consolidated = _metadata_declarations(dimensions, storage_plan)
    manifest = build_refined_detection_run_manifest(
        run_id="refined_manual_1",
        dimensions=dimensions,
        storage_plan=storage_plan,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=2),
        source=RefinedDetectionSourceIdentity(
            run_id="detect_1",
            run_manifest_digest="a" * 64,
            logical_content_digest="b" * 64,
        ),
        instance_reason_codes={0: "none"},
        source_reason_codes={0: "none"},
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )
    arrays = _complete_manual_arrays_same_frame(dimensions)

    assert arrays["instances/frame_row_offsets"].tolist() == [0, 0, 2, 2, 2]
    assert np.unique(arrays["instances/instance_key"]).size == 2
    assert (
        validate_refined_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
        )
        == ()
    )

    invalid = copy.deepcopy(arrays)
    invalid["instances/instance_key"][1] += np.uint64(1)
    assert "manual instance_key values do not match the frozen allocator" in (
        validate_refined_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=invalid,
        )
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
    storage_plan = plan_refined_detection_storage(dimensions)
    direct, consolidated = _metadata_declarations(dimensions, storage_plan)
    manifest = build_refined_detection_run_manifest(
        run_id="refined_clipped_1",
        dimensions=dimensions,
        storage_plan=storage_plan,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
        source=RefinedDetectionSourceCollectionIdentity(
            collection_id="collection_1",
            collection_manifest_digest="1" * 64,
            members=(
                RefinedDetectionClipSourceIdentity(
                    clip_index=0,
                    source_refined_run_id="refined_clip_0",
                    source_refined_manifest_digest="6" * 64,
                    source_detection=RefinedDetectionSourceIdentity(
                        run_id="detect_clip_0",
                        run_manifest_digest="a" * 64,
                        logical_content_digest="b" * 64,
                    ),
                ),
            ),
        ),
        instance_reason_codes={0: "none"},
        source_reason_codes={0: "none"},
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=True,
        clipped_binding=clipped,
    )

    assert validate_refined_detection_run_manifest(manifest) == ()
    binding = manifest["payload"]["logical_schema"]["clipped_binding"]
    assert binding["camera_cardinality"] == 1
    assert binding["clip_ordinal_scope"] == ("snapshot_global_within_single_camera")
    assert binding["empty_frame_media_resolution"] == (
        "complete_frame_map_independent_of_rows"
    )

    tampered = copy.deepcopy(manifest)
    clip = tampered["payload"]["logical_schema"]["clipped_binding"]["clips"][0]
    clip["parent_frame_stop"] = 5
    clip["frame_count"] = 5
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert "clipped_binding intervals must cover logical n_frames" in (
        validate_refined_detection_run_manifest(tampered)
    )

    malformed = copy.deepcopy(manifest)
    malformed["payload"]["logical_schema"]["clipped_binding"]["clips"][0][
        "media_digest"
    ] = "invalid"
    malformed["payload_digest"] = canonical_json_sha256(malformed["payload"])
    assert any(
        "media_digest" in error
        for error in validate_refined_detection_run_manifest(malformed)
    )


def test_clipped_publication_requires_and_checks_bound_clip_artifacts() -> None:
    source_dimensions = RefinedDetectionDimensions(
        n_frames=2,
        n_instances=1,
        n_source_detections=1,
        source_width=640,
        source_height=480,
    )
    source_arrays = _one_raw_clip_arrays(source_dimensions)
    source_plan = plan_refined_detection_storage(source_dimensions)
    source_direct, source_consolidated = _metadata_declarations(
        source_dimensions,
        source_plan,
    )
    raw_source = RefinedDetectionSourceIdentity(
        run_id="detect_clip_0",
        run_manifest_digest="a" * 64,
        logical_content_digest="b" * 64,
    )
    source_manifest = build_refined_detection_run_manifest(
        run_id="refined_clip_0",
        dimensions=source_dimensions,
        storage_plan=source_plan,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=8),
        source=raw_source,
        instance_reason_codes={0: "none"},
        source_reason_codes={0: "none"},
        direct_metadata_declarations=source_direct,
        consolidated_metadata_declarations=source_consolidated,
        selector_eligible=False,
    )
    source_digest = str(source_manifest["payload_digest"])
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
                parent_frame_stop=2,
                frame_map_digest="5" * 64,
                source_refined_run_id="refined_clip_0",
                source_refined_manifest_digest=source_digest,
            ),
        ),
    )
    dimensions = RefinedDetectionDimensions(
        n_frames=2,
        n_instances=1,
        n_source_detections=1,
        source_width=640,
        source_height=480,
        lineage_profile=RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT,
    )
    arrays = _empty_arrays(dimensions)
    arrays.update({path: value.copy() for path, value in source_arrays.items()})
    arrays.update(
        {
            "instances/source_recording_frame_ids": np.asarray([1], dtype=np.int64),
            "instances/source_clip_indices": np.asarray([0], dtype=np.int32),
            "instances/source_clip_local_frame_indices": np.asarray(
                [0], dtype=np.int32
            ),
            "instances/source_clip_detect_row_index": np.asarray([0], dtype=np.int64),
            "instances/source_refined_row_ids": np.asarray([7], dtype=np.int64),
            "source_detections/source_recording_frame_ids": np.asarray(
                [1], dtype=np.int64
            ),
            "source_detections/source_clip_indices": np.asarray([0], dtype=np.int32),
            "source_detections/source_clip_local_frame_indices": np.asarray(
                [0], dtype=np.int32
            ),
            "source_detections/source_clip_detect_row_index": np.asarray(
                [0], dtype=np.int64
            ),
            "source_detections/source_resolved_refined_row_id": np.asarray(
                [7], dtype=np.int64
            ),
        }
    )
    plan = plan_refined_detection_storage(dimensions)
    direct, consolidated = _metadata_declarations(dimensions, plan)
    manifest = build_refined_detection_run_manifest(
        run_id="refined_recording_1",
        dimensions=dimensions,
        storage_plan=plan,
        lineage=_lineage(snapshot_id=NEXT_SNAPSHOT_ID, next_id=9),
        source=RefinedDetectionSourceCollectionIdentity(
            collection_id="collection_1",
            collection_manifest_digest="1" * 64,
            members=(
                RefinedDetectionClipSourceIdentity(
                    clip_index=0,
                    source_refined_run_id="refined_clip_0",
                    source_refined_manifest_digest=source_digest,
                    source_detection=raw_source,
                ),
            ),
        ),
        instance_reason_codes={0: "none"},
        source_reason_codes={0: "none"},
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
        clipped_binding=clipped,
    )
    evidence = (
        RefinedDetectionBoundClipEvidence(
            clip_index=0,
            manifest=source_manifest,
            arrays=source_arrays,
        ),
    )

    assert "clipped publication requires bound per-clip source evidence" in (
        validate_refined_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
        )
    )
    assert (
        validate_refined_detection_publication(
            manifest,
            direct_metadata_declarations=direct,
            consolidated_metadata_declarations=consolidated,
            arrays=arrays,
            clipped_source_evidence=evidence,
        )
        == ()
    )

    invented_identity = copy.deepcopy(arrays)
    invented_identity["instances/refined_row_ids"][0] = 8
    invented_identity["instances/source_refined_row_ids"][0] = 8
    invented_identity["source_detections/resolved_refined_row_id"][0] = 8
    invented_identity["source_detections/source_resolved_refined_row_id"][0] = 8
    assert "clip 0 refined-row membership is incomplete" in (
        validate_refined_detection_clipped_source_evidence(
            manifest,
            invented_identity,
            evidence,
        )
    )

    invented_source_row = copy.deepcopy(arrays)
    invented_source_row["instances/source_clip_detect_row_index"][0] = 5
    invented_source_row["source_detections/source_clip_detect_row_index"][0] = 5
    assert "clip 0 source-audit rows are not complete and ordered" in (
        validate_refined_detection_clipped_source_evidence(
            manifest,
            invented_source_row,
            evidence,
        )
    )


def test_root_snapshot_validates_manual_key_allocator() -> None:
    dimensions = _dimensions(1)
    manifest = _build_manifest(
        run_id="refined_1",
        dimensions=dimensions,
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
    )
    arrays = _manual_arrays([0])

    assert (
        validate_refined_detection_snapshot_identity(
            manifest=manifest,
            arrays=arrays,
        )
        == ()
    )

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

    assert (
        validate_refined_detection_snapshot_identity(
            manifest=current_manifest,
            arrays=current_arrays,
            parent_manifest=parent_manifest,
            parent_arrays=parent_arrays,
        )
        == ()
    )

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

    reused_snapshot = copy.deepcopy(current_manifest)
    reused_snapshot["payload"]["snapshot_lineage"]["snapshot_id"] = ROOT_SNAPSHOT_ID
    reused_snapshot["payload_digest"] = canonical_json_sha256(
        reused_snapshot["payload"]
    )
    assert "successor snapshot_id must differ from parent" in (
        validate_refined_detection_snapshot_identity(
            manifest=reused_snapshot,
            arrays=current_arrays,
            parent_manifest=parent_manifest,
            parent_arrays=parent_arrays,
        )
    )

    changed_recording = copy.deepcopy(current_manifest)
    changed_recording["payload"]["snapshot_lineage"]["manual_instance_key_allocator"][
        "recording_identity"
    ] = "another_recording"
    changed_recording["payload_digest"] = canonical_json_sha256(
        changed_recording["payload"]
    )
    assert "successor recording_identity differs from parent" in (
        validate_refined_detection_snapshot_identity(
            manifest=changed_recording,
            arrays=current_arrays,
            parent_manifest=parent_manifest,
            parent_arrays=parent_arrays,
        )
    )


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


def test_activation_candidate_manifest_stages_final_intent_without_visibility() -> None:
    manifest = _build_manifest(
        run_id="refined_candidate",
        dimensions=_dimensions(1),
        lineage=_lineage(snapshot_id=ROOT_SNAPSHOT_ID, next_id=1),
        selector_eligible=False,
    )

    candidate = build_refined_detection_activation_candidate_manifest(manifest)

    assert manifest["payload"]["publication"]["stage_selector_eligible"] is False
    assert candidate["payload"]["publication"]["stage_selector_eligible"] is True
    assert candidate["payload_digest"] != manifest["payload_digest"]
    assert validate_refined_detection_run_manifest(candidate) == ()
    assert build_refined_detection_activation_candidate_manifest(candidate) == candidate

    invalid = copy.deepcopy(manifest)
    invalid["payload"]["publication"]["completion_status"] = "running"
    invalid["payload_digest"] = canonical_json_sha256(invalid["payload"])
    with pytest.raises(ValueError, match="Cannot prepare an invalid"):
        build_refined_detection_activation_candidate_manifest(invalid)
