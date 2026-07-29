from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from fisheye.shared.zarr.array_factory import array_metadata_declaration_from_plan
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_PERSISTED_PATH,
    CropPixelAuthority,
    CropRefinedSourceIdentity,
    build_crop_row_source_signatures,
    build_crop_run_manifest,
    build_coordinate_crop_run_manifest,
    crop_metadata_declarations_digest,
    normalize_crop_metadata_declarations,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
    CropGeometryPolicy,
    CropPaddingMode,
    CropSizeMode,
    crop_geometry_policy_from_manifest,
    derive_crop_placement_geometry,
    derive_frame_row_offsets,
)
from fisheye.shared.zarr.crop_storage import plan_crop_geometry_storage
from fisheye.shared.zarr.detection_schema import (
    derive_canonical_detection_geometry,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _dimensions() -> CropDimensions:
    return CropDimensions(
        n_frames=4,
        n_instances=4,
        source_width=100,
        source_height=80,
    )


def _policy() -> CropGeometryPolicy:
    return CropGeometryPolicy(
        purpose="subject_analysis",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(8, 8),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
    )


def _source() -> CropRefinedSourceIdentity:
    return CropRefinedSourceIdentity(
        run_id="refined_source",
        run_manifest_digest="a" * 64,
        logical_content_digest="b" * 64,
        recording_identity="crop_manifest_test",
        lineage_id="11111111-1111-4111-8111-111111111111",
        snapshot_id="22222222-2222-4222-8222-222222222222",
    )


def _pixel() -> CropPixelAuthority:
    return CropPixelAuthority(
        authority_id="camera_video_manifest_v1",
        authority_manifest_digest="c" * 64,
        recording_identity="crop_manifest_test",
        camera_identity="cam2010095",
        n_frames=4,
        source_width=100,
        source_height=80,
    )


def _arrays() -> dict[str, np.ndarray]:
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    bbox_norm = np.asarray(
        [
            [0.20, 0.20, 0.10, 0.10],
            [0.70, 0.20, 0.10, 0.10],
            [0.50, 0.70, 0.20, 0.10],
            [0.25, 0.75, 0.10, 0.15],
        ],
        dtype=np.float32,
    )
    bbox_img, centers = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=100,
        source_height=80,
    )
    sizes = np.repeat(np.asarray([[8, 8]], dtype=np.int32), 4, axis=0)
    coordinates, source_crop, bbox_roi = derive_crop_placement_geometry(
        centers,
        bbox_img,
        sizes,
    )
    arrays = {
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_refined_row_ids": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "frame_indices": frames,
        "source_acquisition_frame_index": frames.copy(),
        "frame_row_offsets": derive_frame_row_offsets(frames, n_frames=4),
        "bbox_norm_coords": bbox_norm,
        "bbox_img_xyxy": bbox_img,
        "centers_img_xy": centers,
        "roi_coordinates_full": coordinates,
        "roi_sizes_full": sizes,
        "source_crop_xywh": source_crop,
        "bbox_roi_xyxy": bbox_roi,
    }
    arrays["source_row_signature"] = build_crop_row_source_signatures(
        arrays,
        source=_source(),
        policy=_policy(),
        pixel_authority=_pixel(),
    ).signatures
    return arrays


def _metadata():
    dimensions = _dimensions()
    plan = plan_crop_geometry_storage(dimensions)
    declarations: dict[str, dict[str, object]] = {
        "": {
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "status": "complete",
                "stage_selector_eligible": False,
                "shadow_only": True,
            },
        }
    }
    bindings = {
        binding.path: binding for binding in CROP_GEOMETRY_SCHEMA_V1.bindings
    }
    for entry in plan.entries:
        binding = bindings[entry.rule.path]
        contract = CROP_GEOMETRY_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        declarations[entry.rule.path] = {
            "zarr_format": 3,
            "node_type": "array",
            **array_metadata_declaration_from_plan(
                contract=contract,
                plan=entry.plan,
                fill_value=0,
                attributes={"artifact_class": "geometry_only_analysis"},
            ),
        }
    return plan, declarations, copy.deepcopy(declarations)


def _manifest() -> dict[str, object]:
    plan, direct, consolidated = _metadata()
    return build_crop_run_manifest(
        run_id="crop_shadow",
        dimensions=_dimensions(),
        policy=_policy(),
        storage_plan=plan,
        arrays=_arrays(),
        source=_source(),
        pixel_authority=_pixel(),
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )


def test_manifest_binds_exact_schema_source_pixels_policy_and_signatures() -> None:
    manifest = _manifest()

    assert validate_crop_run_manifest(manifest) == ()
    assert json.loads(json.dumps(manifest)) == manifest
    assert manifest["persisted_path"] == CROP_RUN_MANIFEST_PERSISTED_PATH
    payload = manifest["payload"]
    assert manifest["schema_version"] == 1
    assert "coordinate_contract" not in payload
    assert payload["publication"]["stage_selector_eligible"] is False
    assert payload["publication"]["artifact_class"] == "geometry_only_analysis"
    assert payload["source_refined_snapshot"]["snapshot_id"] == (
        "22222222-2222-4222-8222-222222222222"
    )
    assert payload["source_pixel_authority"]["decoded_pixel_contract"] == {
        "dtype": "uint8",
        "channels": "grayscale",
        "axis_order": "yx",
        "crop_sampling": "integer_half_open_xywh",
    }
    assert payload["row_signature"]["array_path"] == "source_row_signature"
    assert payload["logical_content"]["document"]["arrays"].keys() == set(
        CROP_GEOMETRY_SCHEMA_V1.binding_paths
    )


def test_opt_in_crop_manifest_persists_exact_coordinate_catalog() -> None:
    plan, direct, consolidated = _metadata()
    kwargs = {
        "run_id": "crop_coordinate_shadow",
        "dimensions": _dimensions(),
        "policy": _policy(),
        "storage_plan": plan,
        "arrays": _arrays(),
        "source": _source(),
        "pixel_authority": _pixel(),
        "direct_metadata_declarations": direct,
        "consolidated_metadata_declarations": consolidated,
        "selector_eligible": False,
    }
    legacy = build_crop_run_manifest(**kwargs)
    manifest = build_coordinate_crop_run_manifest(**kwargs)

    assert legacy["schema_version"] == 1
    assert "coordinate_contract" not in legacy["payload"]
    assert manifest["schema_version"] == CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["payload"]["coordinate_contract"]["document"] == (
        CROP_GEOMETRY_SCHEMA_V1.coordinate_contract_manifest()
    )
    assert validate_crop_run_manifest(manifest) == ()

    tampered = copy.deepcopy(manifest)
    catalog = tampered["payload"]["coordinate_contract"]
    catalog["document"]["surfaces"][0]["pixel_convention"] = "wrong"
    catalog["digest"] = canonical_json_sha256(catalog["document"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert "coordinate catalog differs from the frozen stage catalog" in (
        validate_crop_run_manifest(tampered)
    )


def test_policy_parser_rejects_recomputed_digest_semantic_tampering() -> None:
    manifest = _policy().as_manifest()
    assert crop_geometry_policy_from_manifest(manifest) == _policy()

    tampered = copy.deepcopy(manifest)
    tampered["payload"]["placement"]["center_rounding"] = "floor"
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    with pytest.raises(ValueError, match="center rounding"):
        crop_geometry_policy_from_manifest(tampered)


def test_recomputed_outer_digest_cannot_hide_nested_contract_tampering() -> None:
    variants = []
    logical = copy.deepcopy(_manifest())
    logical["payload"]["logical_schema"]["bindings"][0]["required"] = False
    variants.append(logical)

    storage = copy.deepcopy(_manifest())
    storage["payload"]["storage_plan"]["arrays"][0]["plan"]["chunk_shape"][0] = 9
    variants.append(storage)

    source = copy.deepcopy(_manifest())
    source["payload"]["source_refined_snapshot"]["snapshot_id"] = (
        "33333333-3333-4333-8333-333333333333"
    )
    variants.append(source)

    pixel = copy.deepcopy(_manifest())
    pixel["payload"]["source_pixel_authority"]["source_width"] = 101
    variants.append(pixel)

    signature = copy.deepcopy(_manifest())
    signature["payload"]["row_signature"]["spec"]["compatibility_context"][
        "crop_policy_digest"
    ] = "d" * 64
    variants.append(signature)

    content = copy.deepcopy(_manifest())
    content["payload"]["logical_content"]["document"]["arrays"][
        "instance_key"
    ]["shape"] = [3]
    content["payload"]["logical_content"]["digest"] = canonical_json_sha256(
        content["payload"]["logical_content"]["document"]
    )
    variants.append(content)

    unexpected = copy.deepcopy(_manifest())
    unexpected["payload"]["publication"]["unexpected"] = True
    variants.append(unexpected)

    for tampered in variants:
        tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
        assert validate_crop_run_manifest(tampered)


def test_metadata_digest_retains_attributes_and_redacts_only_manifest_cycle() -> None:
    _, direct, consolidated = _metadata()
    before = crop_metadata_declarations_digest(
        direct,
        consolidated_metadata_by_path=consolidated,
        dimensions=_dimensions(),
    )

    with_manifest = copy.deepcopy(direct)
    with_manifest_consolidated = copy.deepcopy(consolidated)
    with_manifest[""]["attributes"]["run_manifest"] = {"circular": True}
    with_manifest_consolidated[""]["attributes"]["run_manifest"] = {
        "circular": True
    }
    assert crop_metadata_declarations_digest(
        with_manifest,
        consolidated_metadata_by_path=with_manifest_consolidated,
        dimensions=_dimensions(),
    ) == before

    changed = copy.deepcopy(direct)
    changed_consolidated = copy.deepcopy(consolidated)
    changed["instance_key"]["attributes"]["new_authoritative_attribute"] = 1
    changed_consolidated["instance_key"]["attributes"][
        "new_authoritative_attribute"
    ] = 1
    assert crop_metadata_declarations_digest(
        changed,
        consolidated_metadata_by_path=changed_consolidated,
        dimensions=_dimensions(),
    ) != before

    divergent = copy.deepcopy(consolidated)
    divergent["instance_key"]["attributes"]["changed"] = True
    with pytest.raises(ValueError, match="Direct and consolidated"):
        normalize_crop_metadata_declarations(
            direct,
            consolidated_metadata_by_path=divergent,
            dimensions=_dimensions(),
        )


def test_metadata_normalizer_accepts_only_exact_empty_group_envelope() -> None:
    _, direct, consolidated = _metadata()
    direct[""]["consolidated_metadata"] = None
    consolidated[""]["consolidated_metadata"] = {
        "kind": "inline",
        "must_understand": False,
        "metadata": {},
    }
    normalize_crop_metadata_declarations(
        direct,
        consolidated_metadata_by_path=consolidated,
        dimensions=_dimensions(),
    )

    malformed = copy.deepcopy(consolidated)
    malformed[""]["consolidated_metadata"]["metadata"] = {"unexpected": {}}
    with pytest.raises(ValueError, match="consolidated_metadata"):
        normalize_crop_metadata_declarations(
            direct,
            consolidated_metadata_by_path=malformed,
            dimensions=_dimensions(),
        )
