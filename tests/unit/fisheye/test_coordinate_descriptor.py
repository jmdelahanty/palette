from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from fisheye.shared import coordinate_descriptor as descriptor_mod
from fisheye.shared.coordinate_identity import (
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
)

from fisheye.shared.coordinate_descriptor import (
    CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION,
    CANONICAL_COORDINATE_PROFILES,
    CANONICAL_OVERLAY_DIRECT,
    CANONICAL_OVERLAY_NOT_SUITABLE,
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
    COORDINATE_DESCRIPTOR_ATTR,
    COORDINATE_DESCRIPTOR_SCHEMA_ID,
    FISH_BODY_FRAME_RECORD_KIND,
    PIXEL_FRAME_AUTHORITY_RECORD_KIND,
    PHYSICAL_FRAME_CALIBRATION_RECORD_KIND,
    CanonicalCollectionAxis,
    CanonicalFrameRecord,
    CoordinateDescriptorError,
    CoordinateRecordRef,
    DigestBoundCoordinateRecordRef,
    LegacySpaceContext,
    build_canonical_coordinate_descriptor,
    build_historical_coordinate_descriptor_v1,
    canonical_coordinate_descriptor_v2_attrs,
    canonical_coordinate_descriptor_v2_digest,
    canonical_coordinate_descriptor_v2_json,
    canonical_coordinate_descriptor_json,
    historical_coordinate_descriptor_v1_attrs,
    coordinate_descriptor_digest,
    load_canonical_coordinate_descriptor_attrs,
    load_coordinate_descriptor_attrs,
    parse_canonical_coordinate_descriptor,
    parse_coordinate_descriptor,
    parse_historical_coordinate_descriptor_v1,
    resolve_legacy_space_id,
    stamp_historical_coordinate_descriptor_v1,
    validate_canonical_coordinate_descriptor,
    validate_coordinate_descriptor,
    verify_canonical_coordinate_descriptor_identity,
)


def _camera_points_descriptor():
    return build_historical_coordinate_descriptor_v1(
        space_id="source_camera_image_px",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        origin="top_left",
        positive_x="right",
        positive_y="down",
        reference_width=4512,
        reference_height=4512,
        reference_units="px",
        reference_authority="/raw_video/images_full.shape[-2:]",
        pixel_convention="pixel_center",
        row_identity_mode="frame_indices",
        row_identity_array_ref="../frame_indices",
        source_camera_overlay="direct",
        lineage_refs=(
            CoordinateRecordRef(
                ref="/provenance/position_source",
                sha256="1" * 64,
            ),
        ),
    )


def _identity_contract(values=(9, 17, 42)):
    return build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=np.asarray(values, dtype=np.uint64),
    )


def _record(ref: str, token: str = "a") -> DigestBoundCoordinateRecordRef:
    return DigestBoundCoordinateRecordRef(
        record_ref=ref,
        record_sha256=token * 64,
    )


def _canonical_camera_descriptor(
    *,
    contract=None,
    identity_ref: str = "/analysis/detect_runs/d1@row_identity_contract",
):
    identity = _identity_contract() if contract is None else contract
    return build_canonical_coordinate_descriptor(
        profile_id="source_camera_image_px.top_left_y_down.v1",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        reference_width=4512,
        reference_height=4512,
        reference_authority=_record(
            "/coordinate_frames/source_camera@pixel_frame_authority",
            "1",
        ),
        reference_selector="record",
        pixel_convention="pixel_center",
        row_identity_contract=identity,
        row_identity_record_ref=identity_ref,
        source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref="/coordinate_frames/source_camera@pixel_frame_authority",
            record_sha256="1" * 64,
        ),
    )


def test_source_camera_unit_vector_profile_is_narrow_and_not_directly_overlayable() -> None:
    identity = _identity_contract()
    frame = _record(
        "/coordinate_frames/source_camera@pixel_frame_authority",
        "1",
    )
    descriptor = build_canonical_coordinate_descriptor(
        profile_id="source_camera_image_px.unit_vector_y_down.v1",
        geometry_type="vector_xy",
        components=("x", "y"),
        component_units=("unitless", "unitless"),
        reference_width=4512,
        reference_height=4512,
        reference_authority=frame,
        reference_selector="record",
        pixel_convention="not_applicable",
        row_identity_contract=identity,
        row_identity_record_ref="/analysis/subject_shape_runs/s1@row_identity_contract",
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame.record_ref,
            record_sha256=frame.record_sha256,
        ),
    )
    assert descriptor.space_id == "source_camera_image_px"
    assert descriptor.origin == "not_applicable"
    assert descriptor.component_units == ("unitless", "unitless")

    payload = descriptor.to_dict()
    payload["source_camera_overlay"]["status"] = CANONICAL_OVERLAY_DIRECT
    assert "profile_overlay_mismatch" in {
        issue.code for issue in validate_canonical_coordinate_descriptor(payload)
    }


def test_source_camera_vector_sequence_and_displacement_profiles_are_distinct() -> None:
    identity = _identity_contract()
    frame = _record(
        "/coordinate_frames/source_camera@pixel_frame_authority",
        "1",
    )
    common = {
        "reference_width": 4512,
        "reference_height": 4512,
        "reference_authority": frame,
        "reference_selector": "record",
        "pixel_convention": "not_applicable",
        "row_identity_contract": identity,
        "row_identity_record_ref": (
            "/analysis/subject_shape_runs/s1@row_identity_contract"
        ),
        "source_camera_overlay_status": CANONICAL_OVERLAY_NOT_SUITABLE,
        "frame_record": CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame.record_ref,
            record_sha256=frame.record_sha256,
        ),
    }
    sequence = build_canonical_coordinate_descriptor(
        profile_id="source_camera_image_px.unit_vector_y_down.v1",
        geometry_type="vector_sequence_xy",
        components=("x", "y"),
        component_units=("unitless", "unitless"),
        **common,
    )
    verify_canonical_coordinate_descriptor_identity(
        sequence,
        row_identity_contract=identity,
        expected_row_identity_record_ref=sequence.row_identity.record_ref,
        owner_shape=(3, 21, 2),
    )
    with pytest.raises(CoordinateDescriptorError):
        verify_canonical_coordinate_descriptor_identity(
            sequence,
            row_identity_contract=identity,
            expected_row_identity_record_ref=sequence.row_identity.record_ref,
            owner_shape=(3, 2),
        )

    displacement = build_canonical_coordinate_descriptor(
        profile_id="source_camera_image_px.displacement_vector_y_down.v1",
        geometry_type="vector_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        **common,
    )
    assert displacement.origin == "not_applicable"
    assert displacement.source_camera_overlay.status == CANONICAL_OVERLAY_NOT_SUITABLE


def _canonical_collected_roi_descriptor(
    *,
    geometry_type: str = "raster_yx",
):
    identity = _identity_contract()
    frame = _record(
        "/analysis/subject_mask_runs/s1/coordinate_frames/roi@pixel_frame_authority",
        "4",
    )
    labels = _record(
        "/analysis/subject_mask_runs/s1@subject_mask_component_labels",
        "5",
    )
    if geometry_type == "raster_yx":
        components = ("y", "x")
        convention = "pixel_center"
    elif geometry_type == "point_xy":
        components = ("x", "y")
        convention = "continuous"
    else:
        components = ("x_min", "y_min", "x_max", "y_max")
        convention = "continuous"
    return build_canonical_coordinate_descriptor(
        profile_id="roi_local_px.top_left_y_down.v1",
        geometry_type=geometry_type,
        components=components,
        component_units=("px",) * len(components),
        reference_width=96,
        reference_height=64,
        reference_authority=frame,
        reference_selector="record",
        pixel_convention=convention,
        row_identity_contract=identity,
        row_identity_record_ref=(
            "/analysis/subject_mask_runs/s1@row_identity_contract"
        ),
        source_camera_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
        overlay_transform_refs=(
            _record(
                "/analysis/subject_mask_runs/s1/source_crop_xywh"
                "@directed_transform_v2",
                "6",
            ),
        ),
        collection_axis=CanonicalCollectionAxis(
            axis=1,
            role="subject_component",
            cardinality=3,
            label_authority=labels,
        ),
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref=frame.record_ref,
            record_sha256=frame.record_sha256,
        ),
    )


def test_round_trip_is_canonical_and_digest_is_deterministic() -> None:
    descriptor = _camera_points_descriptor()
    payload = descriptor.to_dict()

    assert payload["schema_id"] == COORDINATE_DESCRIPTOR_SCHEMA_ID
    assert parse_coordinate_descriptor(payload) == descriptor
    assert parse_coordinate_descriptor(json.dumps(payload)) == descriptor
    assert parse_coordinate_descriptor(json.dumps(payload).encode("utf-8")) == descriptor
    assert descriptor.digest() == coordinate_descriptor_digest(payload)
    assert descriptor.canonical_json() == canonical_coordinate_descriptor_json(payload)

    reordered = {key: payload[key] for key in reversed(tuple(payload))}
    assert coordinate_descriptor_digest(reordered) == descriptor.digest()


def test_schema_versions_require_exact_json_integers() -> None:
    historical = _camera_points_descriptor().to_dict()
    historical["schema_version"] = 1.0
    with pytest.raises(CoordinateDescriptorError):
        parse_coordinate_descriptor(historical)

    canonical = _canonical_camera_descriptor().to_dict()
    canonical["schema_version"] = 2.0
    with pytest.raises(CoordinateDescriptorError):
        parse_canonical_coordinate_descriptor(canonical)


def test_historical_json_rejects_duplicate_keys_recursively() -> None:
    raw = _camera_points_descriptor().canonical_json().replace(
        '"space_id":"source_camera_image_px"',
        '"space_id":"roi_local_px","space_id":"source_camera_image_px"',
        1,
    )
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        parse_historical_coordinate_descriptor_v1(raw)
    assert {issue.code for issue in exc_info.value.issues} == {
        "descriptor_json_invalid"
    }


def test_attrs_are_compact_and_detect_tampering() -> None:
    descriptor = _camera_points_descriptor()
    attrs = historical_coordinate_descriptor_v1_attrs(descriptor)

    assert set(attrs) == {
        COORDINATE_DESCRIPTOR_ATTR,
        f"{COORDINATE_DESCRIPTOR_ATTR}_sha256",
    }
    assert load_coordinate_descriptor_attrs(attrs) == descriptor

    tampered = copy.deepcopy(attrs)
    tampered[COORDINATE_DESCRIPTOR_ATTR]["reference_extent"]["width"] = 640
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        load_coordinate_descriptor_attrs(tampered)
    assert {issue.code for issue in exc_info.value.issues} == {
        "descriptor_digest_mismatch"
    }


def test_stamp_helper_writes_only_descriptor_and_digest() -> None:
    class Node:
        def __init__(self) -> None:
            self.attrs: dict[str, object] = {"existing": True}

    node = Node()
    descriptor = _camera_points_descriptor()
    assert (
        stamp_historical_coordinate_descriptor_v1(node, descriptor)
        == descriptor
    )
    assert node.attrs["existing"] is True
    assert load_coordinate_descriptor_attrs(node.attrs) == descriptor


def test_lineage_and_transform_provenance_remain_separate_references() -> None:
    descriptor = build_historical_coordinate_descriptor_v1(
        space_id="roi_local_px",
        geometry_type="bbox_xyxy",
        components=("x_min", "y_min", "x_max", "y_max"),
        component_units=("px", "px", "px", "px"),
        origin="top_left",
        positive_x="right",
        positive_y="down",
        reference_width=512,
        reference_height=512,
        reference_units="px",
        reference_authority="/crop_runs/crop_1/roi_images.shape[-2:]",
        pixel_convention="pixel_edge_half_open",
        row_identity_mode="source_crop_row_ids",
        row_identity_array_ref="../source_crop_row_ids",
        source_camera_overlay="requires_transform",
        transform_refs=(
            CoordinateRecordRef(
                ref="/coordinate_records/crop_1/source_image_to_roi",
                sha256="a" * 64,
            ),
        ),
        lineage_refs=(
            CoordinateRecordRef(
                ref="/coordinate_records/crop_1/source_selection",
                sha256="b" * 64,
            ),
        ),
    )

    payload = descriptor.to_dict()
    assert payload["transform_refs"] == [
        {
            "ref": "/coordinate_records/crop_1/source_image_to_roi",
            "sha256": "a" * 64,
        }
    ]
    assert payload["lineage_refs"] == [
        {
            "ref": "/coordinate_records/crop_1/source_selection",
            "sha256": "b" * 64,
        }
    ]
    assert "transform" not in payload
    assert "lineage" not in payload


def test_structured_issues_report_unknown_missing_and_component_errors() -> None:
    payload = _camera_points_descriptor().to_dict()
    del payload["origin"]
    payload["guessed_space"] = "camera"
    payload["component_units"] = ["px"]

    issues = validate_coordinate_descriptor(payload)
    codes = {issue.code for issue in issues}
    assert "missing_field" in codes
    assert "unknown_field" in codes
    assert "component_unit_count_mismatch" in codes

    with pytest.raises(CoordinateDescriptorError) as exc_info:
        parse_coordinate_descriptor(payload)
    assert exc_info.value.issues == issues


@pytest.mark.parametrize(
    "space_id",
    (
        "crimson_viewport_px",
        "display_px",
        "renderer_overlay_px",
        "screen_pixels",
    ),
)
def test_presentation_spaces_are_forbidden(space_id: str) -> None:
    payload = _camera_points_descriptor().to_dict()
    payload["space_id"] = space_id
    payload["source_camera_overlay"] = "requires_transform"

    codes = {issue.code for issue in validate_coordinate_descriptor(payload)}
    assert "presentation_space_forbidden" in codes


def test_physical_mm_requires_named_physical_frame() -> None:
    kwargs = dict(
        space_id="physical_mm",
        geometry_type="point_xy",
        components=("x", "y"),
        component_units=("mm", "mm"),
        origin="physical_frame_origin",
        positive_x="right",
        positive_y="up",
        reference_width=None,
        reference_height=None,
        reference_units="not_applicable",
        reference_authority="/coordinate_records/arena_physical_frame",
        pixel_convention="not_applicable",
        row_identity_mode="sample_indices",
        row_identity_array_ref="../sample_indices",
        source_camera_overlay="requires_transform",
    )
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        build_historical_coordinate_descriptor_v1(**kwargs)
    assert "physical_frame_required" in {
        issue.code for issue in exc_info.value.issues
    }

    descriptor = build_historical_coordinate_descriptor_v1(
        **kwargs,
        physical_frame="arena_1_mm",
    )
    assert descriptor.physical_frame == "arena_1_mm"


def test_pixel_spaces_require_authoritative_positive_reference_dimensions() -> None:
    payload = _camera_points_descriptor().to_dict()
    payload["reference_extent"]["width"] = None
    payload["reference_extent"]["height"] = None
    codes = {issue.code for issue in validate_coordinate_descriptor(payload)}
    assert "reference_extent_required" in codes

    payload = _camera_points_descriptor().to_dict()
    payload["reference_extent"]["units"] = "mm"
    codes = {issue.code for issue in validate_coordinate_descriptor(payload)}
    assert "reference_units_inconsistent" in codes


def test_row_identity_requires_an_explicit_array_reference() -> None:
    payload = _camera_points_descriptor().to_dict()
    payload["row_identity"]["array_ref"] = None
    codes = {issue.code for issue in validate_coordinate_descriptor(payload)}
    assert "row_identity_ref_required" in codes

    payload["row_identity"] = {"mode": "not_applicable", "array_ref": None}
    assert not validate_coordinate_descriptor(payload)


def test_direct_overlay_is_reserved_for_source_camera_image_pixels() -> None:
    payload = _camera_points_descriptor().to_dict()
    payload["space_id"] = "roi_local_px"
    codes = {issue.code for issue in validate_coordinate_descriptor(payload)}
    assert "overlay_status_inconsistent" in codes

    camera_payload = _camera_points_descriptor().to_dict()
    camera_payload["source_camera_overlay"] = "not_suitable"
    assert not validate_coordinate_descriptor(camera_payload)


@pytest.mark.parametrize(
    ("geometry_type", "components"),
    (
        ("bbox_xywh", ("x", "y", "width", "height")),
        ("bbox_cxcywh", ("center_x", "center_y", "width", "height")),
    ),
)
def test_current_bbox_layouts_have_explicit_component_contracts(
    geometry_type: str,
    components: tuple[str, ...],
) -> None:
    descriptor = build_historical_coordinate_descriptor_v1(
        space_id="detector_normalized_xy",
        geometry_type=geometry_type,
        components=components,
        component_units=("normalized",) * 4,
        origin="top_left",
        positive_x="right",
        positive_y="down",
        reference_width=640,
        reference_height=640,
        reference_units="px",
        reference_authority="/raw_video/images_ds.shape[-2:]",
        pixel_convention="continuous",
        row_identity_mode="frame_indices",
        row_identity_array_ref="../frame_indices",
        source_camera_overlay="requires_transform",
    )
    assert descriptor.components == components


def test_legacy_labels_fail_closed_without_explicit_context() -> None:
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        resolve_legacy_space_id("camera", context=None)
    assert {issue.code for issue in exc_info.value.issues} == {
        "legacy_context_missing"
    }

    insufficient = LegacySpaceContext(
        canonical_space_id="source_camera_image_px",
        reference_width=4512,
        reference_height=4512,
        reference_units="px",
        reference_authority="/raw_video/images_full.shape[-2:]",
        evidence_refs=(),
    )
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        resolve_legacy_space_id("camera", context=insufficient)
    assert "legacy_context_evidence_missing" in {
        issue.code for issue in exc_info.value.issues
    }


def test_persisted_legacy_label_must_match_space_and_cite_evidence() -> None:
    payload = _camera_points_descriptor().to_dict()
    payload["legacy_space_label"] = "texture"
    codes = {issue.code for issue in validate_coordinate_descriptor(payload)}
    assert "legacy_label_space_mismatch" in codes

    payload["legacy_space_label"] = "camera"
    del payload["lineage_refs"]
    codes = {issue.code for issue in validate_coordinate_descriptor(payload)}
    assert "legacy_label_evidence_required" in codes


@pytest.mark.parametrize(
    ("legacy_label", "canonical_space_id"),
    (
        ("camera", "source_camera_image_px"),
        ("texture", "stimulus_texture_px"),
    ),
)
def test_legacy_labels_resolve_only_with_matching_authoritative_evidence(
    legacy_label: str,
    canonical_space_id: str,
) -> None:
    context = LegacySpaceContext(
        canonical_space_id=canonical_space_id,
        reference_width=4512 if legacy_label == "camera" else 358,
        reference_height=4512 if legacy_label == "camera" else 358,
        reference_units="px",
        reference_authority="/coordinate_records/legacy_reference_extent",
        evidence_refs=(
            CoordinateRecordRef(
                ref="/provenance/selected_position_source",
                sha256="b" * 64,
            ),
        ),
    )
    assert resolve_legacy_space_id(legacy_label, context=context) == canonical_space_id

    wrong = LegacySpaceContext(
        canonical_space_id="stimulus_texture_px"
        if canonical_space_id == "source_camera_image_px"
        else "source_camera_image_px",
        reference_width=context.reference_width,
        reference_height=context.reference_height,
        reference_units="px",
        reference_authority=context.reference_authority,
        evidence_refs=context.evidence_refs,
    )
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        resolve_legacy_space_id(legacy_label, context=wrong)
    assert "legacy_context_space_mismatch" in {
        issue.code for issue in exc_info.value.issues
    }


def test_canonical_v2_round_trip_is_compact_digest_bound_and_deterministic() -> None:
    descriptor = _canonical_camera_descriptor()
    payload = descriptor.to_dict()

    assert payload["schema_id"] == COORDINATE_DESCRIPTOR_SCHEMA_ID
    assert payload["schema_version"] == CANONICAL_COORDINATE_DESCRIPTOR_SCHEMA_VERSION
    assert set(payload["row_identity"]) == {"record_ref", "record_sha256"}
    assert set(payload["reference_extent"]["authority"]) == {
        "record_ref",
        "record_sha256",
        "selector",
    }
    assert payload["lineage_refs"] == [
        {
            "record_ref": "/coordinate_frames/source_camera@pixel_frame_authority",
            "record_sha256": "1" * 64,
        }
    ]
    assert parse_canonical_coordinate_descriptor(payload) == descriptor
    assert parse_canonical_coordinate_descriptor(descriptor.canonical_json()) == descriptor
    assert parse_canonical_coordinate_descriptor(
        descriptor.canonical_json().encode("utf-8")
    ) == descriptor

    reordered = {name: payload[name] for name in reversed(tuple(payload))}
    assert canonical_coordinate_descriptor_v2_digest(reordered) == descriptor.digest()
    assert canonical_coordinate_descriptor_v2_json(reordered) == descriptor.canonical_json()


def test_canonical_pixel_reference_extent_requires_exact_integers() -> None:
    payload = _canonical_camera_descriptor().to_dict()
    payload["reference_extent"]["width"] = 4512.5
    payload["reference_extent"]["height"] = 999.25

    assert "pixel_reference_extent_not_integer" in {
        issue.code for issue in validate_canonical_coordinate_descriptor(payload)
    }


def test_historical_v1_and_canonical_v2_have_explicit_non_dispatching_apis() -> None:
    historical = _camera_points_descriptor()
    assert parse_historical_coordinate_descriptor_v1(historical.to_dict()) == historical

    with pytest.raises(CoordinateDescriptorError) as exc_info:
        parse_canonical_coordinate_descriptor(historical)
    assert "canonical_schema_version_required" in {
        issue.code for issue in exc_info.value.issues
    }

    with pytest.raises(CoordinateDescriptorError):
        build_canonical_coordinate_descriptor(
            profile_id="source_camera_image_px.top_left_y_down.v1",
            geometry_type="points_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            reference_width=4512,
            reference_height=4512,
            reference_authority=_record(
                "/raw_video/images_full@zarr_metadata",
                "1",
            ),
            reference_selector="shape[-2:]",
            pixel_convention="pixel_center",
            row_identity_contract=historical,
            row_identity_record_ref=(
                "/analysis/detect_runs/d1@row_identity_contract"
            ),
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        )


@pytest.mark.parametrize(
    ("surface", "expected_code"),
    [
        ("identity", "record_digest_required"),
        ("authority", "record_digest_required"),
        ("lineage", "record_digest_required"),
    ],
)
def test_canonical_v2_rejects_missing_identity_authority_and_lineage_digests(
    surface: str,
    expected_code: str,
) -> None:
    payload = _canonical_camera_descriptor().to_dict()
    if surface == "identity":
        del payload["row_identity"]["record_sha256"]
    elif surface == "authority":
        del payload["reference_extent"]["authority"]["record_sha256"]
    else:
        del payload["lineage_refs"][0]["record_sha256"]

    codes = {
        issue.code for issue in validate_canonical_coordinate_descriptor(payload)
    }
    assert expected_code in codes


def test_transform_required_overlay_names_an_ordered_digest_bound_chain() -> None:
    contract = _identity_contract()
    kwargs = dict(
        profile_id="roi_local_px.top_left_y_down.v1",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        reference_width=512,
        reference_height=512,
        reference_authority=_record(
            "/analysis/crop_runs/c1@roi_pixel_frame_authority",
            "2",
        ),
        reference_selector="record",
        pixel_convention="pixel_center",
        row_identity_contract=contract,
        row_identity_record_ref="/analysis/crop_runs/c1@row_identity_contract",
        source_camera_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref="/analysis/crop_runs/c1@roi_pixel_frame_authority",
            record_sha256="2" * 64,
        ),
    )
    with pytest.raises(CoordinateDescriptorError) as missing:
        build_canonical_coordinate_descriptor(**kwargs)
    assert "record_refs_invalid" in {issue.code for issue in missing.value.issues}

    descriptor = build_canonical_coordinate_descriptor(
        **kwargs,
        overlay_transform_refs=(
            _record(
                "/analysis/crop_runs/c1@roi_to_source_image_transform",
                "3",
            ),
        ),
    )
    overlay = descriptor.to_dict()["source_camera_overlay"]
    assert overlay["chain_direction"] == "descriptor_to_source_camera_image"
    assert overlay["transform_refs"][0]["record_sha256"] == "3" * 64

    payload = descriptor.to_dict()
    del payload["source_camera_overlay"]["transform_refs"][0]["record_sha256"]
    assert "record_digest_required" in {
        issue.code for issue in validate_canonical_coordinate_descriptor(payload)
    }


def test_direct_overlay_is_only_available_to_source_camera_image_profile() -> None:
    contract = _identity_contract()
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        build_canonical_coordinate_descriptor(
            profile_id="roi_local_px.top_left_y_down.v1",
            geometry_type="points_xy",
            components=("x", "y"),
            component_units=("px", "px"),
            reference_width=512,
            reference_height=512,
            reference_authority=_record(
                "/analysis/crop_runs/c1/roi_images@zarr_metadata",
                "2",
            ),
            reference_selector="shape[-2:]",
            pixel_convention="pixel_center",
            row_identity_contract=contract,
            row_identity_record_ref="/analysis/crop_runs/c1@row_identity_contract",
            source_camera_overlay_status=CANONICAL_OVERLAY_DIRECT,
        )
    assert "profile_overlay_mismatch" in {
        issue.code for issue in exc_info.value.issues
    }


def test_physical_mm_requires_exact_digest_bound_frame_calibration_record() -> None:
    contract = _identity_contract()
    authority = _record(
        "/analysis/calibration/arena_1@physical_frame_calibration",
        "4",
    )
    kwargs = dict(
        profile_id="physical_mm.source_camera_y_down.v1",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("mm", "mm"),
        reference_width=100.0,
        reference_height=100.0,
        reference_authority=authority,
        reference_selector="record",
        pixel_convention="not_applicable",
        row_identity_contract=contract,
        row_identity_record_ref="/analysis/tracks/t1@row_identity_contract",
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
    )
    with pytest.raises(CoordinateDescriptorError) as missing:
        build_canonical_coordinate_descriptor(**kwargs)
    assert "frame_record_required" in {issue.code for issue in missing.value.issues}

    mismatched_frame = CanonicalFrameRecord(
        kind=PHYSICAL_FRAME_CALIBRATION_RECORD_KIND,
        record_ref="/analysis/calibration/other@physical_frame_calibration",
        record_sha256="5" * 64,
    )
    with pytest.raises(CoordinateDescriptorError) as mismatch:
        build_canonical_coordinate_descriptor(
            **kwargs,
            frame_record=mismatched_frame,
        )
    assert "frame_authority_mismatch" in {
        issue.code for issue in mismatch.value.issues
    }

    descriptor = build_canonical_coordinate_descriptor(
        **kwargs,
        frame_record=CanonicalFrameRecord(
            kind=PHYSICAL_FRAME_CALIBRATION_RECORD_KIND,
            record_ref=authority.record_ref,
            record_sha256=authority.record_sha256,
        ),
    )
    assert descriptor.frame_record is not None
    assert descriptor.frame_record.kind == PHYSICAL_FRAME_CALIBRATION_RECORD_KIND


@pytest.mark.parametrize(
    "profile_id",
    (
        "physical_mm.arena_y_down.v1",
        "physical_mm.cartesian_y_up.v1",
    ),
)
def test_unsupported_physical_profiles_are_reserved_for_future_transforms(
    profile_id: str,
) -> None:
    assert CANONICAL_COORDINATE_PROFILES[profile_id].publication_status != "available"
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        build_canonical_coordinate_descriptor(
            profile_id=profile_id,
            geometry_type="points_xy",
            components=("x", "y"),
            component_units=("mm", "mm"),
            reference_width=None,
            reference_height=None,
            reference_authority=_record(
                "/analysis/calibration/reserved@physical_frame_calibration",
                "4",
            ),
            reference_selector="record",
            pixel_convention="not_applicable",
            row_identity_contract=_identity_contract(),
            row_identity_record_ref="/analysis/tracks/t1@row_identity_contract",
            source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        )
    assert {issue.code for issue in exc_info.value.issues} == {
        "profile_publication_unavailable"
    }


def test_fish_body_coordinates_require_the_exact_body_frame_record() -> None:
    contract = _identity_contract()
    body = _record(
        "/analysis/subject_shape_runs/s1@body_frame_contract",
        "6",
    )
    descriptor = build_canonical_coordinate_descriptor(
        profile_id="fish_anatomical_body_frame.px_anterior_left.v1",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        reference_width=None,
        reference_height=None,
        reference_authority=body,
        reference_selector="record",
        pixel_convention="not_applicable",
        row_identity_contract=contract,
        row_identity_record_ref=(
            "/analysis/subject_shape_runs/s1@row_identity_contract"
        ),
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        frame_record=CanonicalFrameRecord(
            kind=FISH_BODY_FRAME_RECORD_KIND,
            record_ref=body.record_ref,
            record_sha256=body.record_sha256,
        ),
    )
    assert descriptor.positive_directions.x == "anterior"
    assert descriptor.positive_directions.y == "anatomical_left"

    payload = descriptor.to_dict()
    payload["frame_record"]["record_sha256"] = "7" * 64
    assert {
        "frame_record_lineage_missing",
        "frame_authority_mismatch",
    }.issubset(
        {
            issue.code
            for issue in validate_canonical_coordinate_descriptor(payload)
        }
    )


def test_px_and_mm_siblings_share_one_external_row_identity_record_exactly() -> None:
    contract = _identity_contract()
    identity_ref = "/analysis/tracks/t1@row_identity_contract"
    pixels = _canonical_camera_descriptor(
        contract=contract,
        identity_ref=identity_ref,
    )
    physical = _record(
        "/analysis/calibration/arena_1@physical_frame_calibration",
        "8",
    )
    millimetres = build_canonical_coordinate_descriptor(
        profile_id="physical_mm.source_camera_y_down.v1",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("mm", "mm"),
        reference_width=100.0,
        reference_height=100.0,
        reference_authority=physical,
        reference_selector="record",
        pixel_convention="not_applicable",
        row_identity_contract=contract,
        row_identity_record_ref=identity_ref,
        source_camera_overlay_status=CANONICAL_OVERLAY_NOT_SUITABLE,
        frame_record=CanonicalFrameRecord(
            kind=PHYSICAL_FRAME_CALIBRATION_RECORD_KIND,
            record_ref=physical.record_ref,
            record_sha256=physical.record_sha256,
        ),
    )

    assert pixels.row_identity == millimetres.row_identity
    assert pixels.to_dict()["row_identity"] == millimetres.to_dict()["row_identity"]
    assert pixels.row_identity.record_sha256 == contract.digest()


def test_canonical_unbound_attrs_load_verifies_identity_ref_and_row_count() -> None:
    contract = _identity_contract()
    identity_ref = "/analysis/detect_runs/d1@row_identity_contract"
    descriptor = _canonical_camera_descriptor(
        contract=contract,
        identity_ref=identity_ref,
    )

    class Node:
        def __init__(self, rows: int) -> None:
            self.shape = (rows, 2)
            self.attrs: dict[str, object] = {"keep": True}

    node = Node(contract.leading_dimension)
    node.attrs.update(canonical_coordinate_descriptor_v2_attrs(descriptor))
    assert node.attrs["keep"] is True
    assert load_canonical_coordinate_descriptor_attrs(
        node.attrs,
        row_identity_contract=contract,
        expected_row_identity_record_ref=identity_ref,
        owner_shape=node.shape,
    ) == descriptor

    with pytest.raises(CoordinateDescriptorError) as count_error:
        load_canonical_coordinate_descriptor_attrs(
            node.attrs,
            row_identity_contract=contract,
            expected_row_identity_record_ref=identity_ref,
            owner_shape=(contract.leading_dimension - 1, 2),
        )
    assert "row_identity_count_mismatch" in {
        issue.code for issue in count_error.value.issues
    }

    different_contract = _identity_contract((9, 17, 99))
    with pytest.raises(CoordinateDescriptorError) as digest_error:
        load_canonical_coordinate_descriptor_attrs(
            node.attrs,
            row_identity_contract=different_contract,
            expected_row_identity_record_ref=identity_ref,
            owner_shape=node.shape,
        )
    assert "row_identity_record_digest_mismatch" in {
        issue.code for issue in digest_error.value.issues
    }

    wrong_layout = Node(contract.leading_dimension)
    wrong_layout.shape = (contract.leading_dimension, 99)
    with pytest.raises(CoordinateDescriptorError) as layout_error:
        verify_canonical_coordinate_descriptor_identity(
            descriptor,
            row_identity_contract=contract,
            expected_row_identity_record_ref=identity_ref,
            owner_shape=wrong_layout.shape,
        )
    assert "geometry_owner_shape_mismatch" in {
        issue.code for issue in layout_error.value.issues
    }


@pytest.mark.parametrize(
    ("geometry_type", "owner_shape"),
    (
        ("raster_yx", (3, 3, 64, 96)),
        ("point_xy", (3, 3, 2)),
        ("bbox_xyxy", (3, 3, 4)),
    ),
)
def test_subject_component_collection_axis_round_trips_and_validates_shape(
    geometry_type: str,
    owner_shape: tuple[int, ...],
) -> None:
    descriptor = _canonical_collected_roi_descriptor(
        geometry_type=geometry_type,
    )
    payload = descriptor.to_dict()

    assert parse_canonical_coordinate_descriptor(payload) == descriptor
    assert canonical_coordinate_descriptor_v2_digest(payload) == descriptor.digest()
    assert payload["collection_axis"] == {
        "axis": 1,
        "role": "subject_component",
        "cardinality": 3,
        "label_authority": {
            "record_ref": (
                "/analysis/subject_mask_runs/s1"
                "@subject_mask_component_labels"
            ),
            "record_sha256": "5" * 64,
        },
    }
    verify_canonical_coordinate_descriptor_identity(
        descriptor,
        row_identity_contract=_identity_contract(),
        expected_row_identity_record_ref=(
            "/analysis/subject_mask_runs/s1@row_identity_contract"
        ),
        owner_shape=owner_shape,
    )


@pytest.mark.parametrize(
    ("geometry_type", "owner_shape", "valid"),
    (
        ("point_xy", (3, 2), True),
        ("point_xy", (3, 2, 2), False),
        ("points_xy", (3, 2, 2), True),
        ("points_xy", (3, 2), False),
    ),
)
def test_point_and_points_geometry_require_distinct_physical_layouts(
    geometry_type: str,
    owner_shape: tuple[int, ...],
    valid: bool,
) -> None:
    payload = _canonical_camera_descriptor().to_dict()
    payload["geometry_type"] = geometry_type
    descriptor = parse_canonical_coordinate_descriptor(payload)
    kwargs = {
        "row_identity_contract": _identity_contract(),
        "expected_row_identity_record_ref": descriptor.row_identity.record_ref,
        "owner_shape": owner_shape,
    }

    if valid:
        verify_canonical_coordinate_descriptor_identity(descriptor, **kwargs)
    else:
        with pytest.raises(CoordinateDescriptorError) as exc_info:
            verify_canonical_coordinate_descriptor_identity(descriptor, **kwargs)
        assert "geometry_owner_shape_mismatch" in {
            issue.code for issue in exc_info.value.issues
        }


@pytest.mark.parametrize(
    ("geometry_type", "owner_shape", "issue_code"),
    (
        ("raster_yx", (3, 2, 64, 96), "collection_axis_cardinality_mismatch"),
        ("raster_yx", (3, 3, 64), "geometry_owner_shape_mismatch"),
        ("point_xy", (3, 3, 4), "geometry_owner_shape_mismatch"),
        ("bbox_xyxy", (3, 3, 2), "geometry_owner_shape_mismatch"),
    ),
)
def test_subject_component_collection_axis_rejects_wrong_physical_layout(
    geometry_type: str,
    owner_shape: tuple[int, ...],
    issue_code: str,
) -> None:
    descriptor = _canonical_collected_roi_descriptor(
        geometry_type=geometry_type,
    )
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        verify_canonical_coordinate_descriptor_identity(
            descriptor,
            row_identity_contract=_identity_contract(),
            expected_row_identity_record_ref=(
                "/analysis/subject_mask_runs/s1@row_identity_contract"
            ),
            owner_shape=owner_shape,
        )
    assert issue_code in {issue.code for issue in exc_info.value.issues}


def test_subject_component_collection_axis_rejects_authority_and_axis_tampering() -> None:
    payload = _canonical_collected_roi_descriptor().to_dict()
    payload["collection_axis"]["label_authority"]["record_sha256"] = "7" * 64
    assert "collection_axis_authority_lineage_missing" in {
        issue.code for issue in validate_canonical_coordinate_descriptor(payload)
    }

    payload = _canonical_collected_roi_descriptor().to_dict()
    payload["collection_axis"]["axis"] = 2
    assert "collection_axis_index_unsupported" in {
        issue.code for issue in validate_canonical_coordinate_descriptor(payload)
    }


@pytest.mark.parametrize(
    "role",
    ["subject_component", "keypoint", "tail_segment", "chaser"],
)
def test_controlled_collection_axis_roles_are_explicitly_supported(role: str) -> None:
    payload = _canonical_collected_roi_descriptor().to_dict()
    payload["collection_axis"]["role"] = role

    parsed = parse_canonical_coordinate_descriptor(payload)

    assert parsed.collection_axis is not None
    assert parsed.collection_axis.role == role


def test_canonical_v2_without_collection_axis_remains_byte_compatible() -> None:
    descriptor = _canonical_camera_descriptor()
    payload = descriptor.to_dict()

    assert "collection_axis" not in payload
    assert canonical_coordinate_descriptor_v2_json(payload) == descriptor.canonical_json()
    assert canonical_coordinate_descriptor_v2_digest(payload) == descriptor.digest()


def test_canonical_descriptor_module_exposes_no_unsealed_write_bypass() -> None:
    assert "stamp_canonical_coordinate_descriptor" not in descriptor_mod.__all__
    assert not hasattr(descriptor_mod, "stamp_canonical_coordinate_descriptor")
    for generic_v1_writer in (
        "build_coordinate_descriptor",
        "coordinate_descriptor_attrs",
        "stamp_coordinate_descriptor",
    ):
        assert generic_v1_writer not in descriptor_mod.__all__
        assert not hasattr(descriptor_mod, generic_v1_writer)


def test_canonical_attrs_detect_descriptor_tampering() -> None:
    descriptor = _canonical_camera_descriptor()
    attrs = canonical_coordinate_descriptor_v2_attrs(descriptor)
    attrs[COORDINATE_DESCRIPTOR_ATTR]["reference_extent"]["width"] = 640

    with pytest.raises(CoordinateDescriptorError) as exc_info:
        load_canonical_coordinate_descriptor_attrs(
            attrs,
            row_identity_contract=_identity_contract(),
            expected_row_identity_record_ref=(
                "/analysis/detect_runs/d1@row_identity_contract"
            ),
            owner_shape=(3, 2),
        )
    assert "descriptor_digest_mismatch" in {
        issue.code for issue in exc_info.value.issues
    }


def test_canonical_attrs_require_exact_persisted_mapping_form() -> None:
    descriptor = _canonical_camera_descriptor()
    attrs = canonical_coordinate_descriptor_v2_attrs(descriptor)

    class MappingSubclass(dict[str, object]):
        pass

    attrs[COORDINATE_DESCRIPTOR_ATTR] = MappingSubclass(
        attrs[COORDINATE_DESCRIPTOR_ATTR]
    )
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        load_canonical_coordinate_descriptor_attrs(
            attrs,
            row_identity_contract=_identity_contract(),
            expected_row_identity_record_ref=(
                "/analysis/detect_runs/d1@row_identity_contract"
            ),
            owner_shape=(3, 2),
        )
    assert {issue.code for issue in exc_info.value.issues} == {
        "descriptor_persisted_form_noncanonical"
    }


@pytest.mark.parametrize(
    "space_id",
    ("crimson_viewport_px", "display_px", "renderer_overlay_px"),
)
def test_canonical_viewport_and_presentation_spaces_fail_closed(
    space_id: str,
) -> None:
    payload = _canonical_camera_descriptor().to_dict()
    payload["space_id"] = space_id
    assert "presentation_space_forbidden" in {
        issue.code for issue in validate_canonical_coordinate_descriptor(payload)
    }


def test_profile_fields_cannot_be_recombined_into_impossible_semantics() -> None:
    payload = _canonical_camera_descriptor().to_dict()
    payload["component_units"] = ["mm", "mm"]
    payload["origin"] = "physical_frame_origin"
    payload["positive_directions"]["y"] = "up"
    payload["reference_extent"]["units"] = "mm"

    codes = {
        issue.code for issue in validate_canonical_coordinate_descriptor(payload)
    }
    assert "profile_component_unit_mismatch" in codes
    assert "profile_field_mismatch" in codes

    bbox = _canonical_camera_descriptor().to_dict()
    bbox["geometry_type"] = "bbox_xyxy"
    bbox["components"] = ["x_min", "y_min", "x_max", "y_max"]
    bbox["component_units"] = ["px", "px", "px", "px"]
    assert "geometry_pixel_convention_mismatch" in {
        issue.code for issue in validate_canonical_coordinate_descriptor(bbox)
    }


def test_roi_and_image_numeric_overlap_never_changes_declared_semantics() -> None:
    coordinates = np.asarray([[10.0, 20.0], [100.0, 200.0]])
    contract = build_row_identity_contract(
        domain=OBSERVATION_INSTANCE_DOMAIN,
        values=np.asarray([1, 2], dtype=np.uint64),
    )
    image = _canonical_camera_descriptor(contract=contract)
    roi = build_canonical_coordinate_descriptor(
        profile_id="roi_local_px.top_left_y_down.v1",
        geometry_type="points_xy",
        components=("x", "y"),
        component_units=("px", "px"),
        reference_width=4512,
        reference_height=4512,
        reference_authority=_record(
            "/analysis/crop_runs/c1@roi_pixel_frame_authority",
            "9",
        ),
        reference_selector="record",
        pixel_convention="pixel_center",
        row_identity_contract=contract,
        row_identity_record_ref="/analysis/detect_runs/d1@row_identity_contract",
        source_camera_overlay_status=CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
        overlay_transform_refs=(
            _record("/analysis/crop_runs/c1@roi_to_source_image_transform", "a"),
        ),
        frame_record=CanonicalFrameRecord(
            kind=PIXEL_FRAME_AUTHORITY_RECORD_KIND,
            record_ref="/analysis/crop_runs/c1@roi_pixel_frame_authority",
            record_sha256="9" * 64,
        ),
    )

    assert coordinates.shape[0] == contract.leading_dimension
    assert image.space_id == "source_camera_image_px"
    assert roi.space_id == "roi_local_px"
    assert image.digest() != roi.digest()


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            '"space_id":"source_camera_image_px"',
            '"space_id":"roi_local_px","space_id":"source_camera_image_px"',
        ),
        (
            '"selector":"record"',
            '"selector":"shape[-2:]","selector":"record"',
        ),
    ],
)
def test_canonical_json_rejects_duplicate_keys_recursively(
    needle: str,
    replacement: str,
) -> None:
    raw = _canonical_camera_descriptor().canonical_json().replace(
        needle,
        replacement,
        1,
    )
    with pytest.raises(CoordinateDescriptorError) as exc_info:
        parse_canonical_coordinate_descriptor(raw)
    assert {issue.code for issue in exc_info.value.issues} == {
        "descriptor_json_invalid"
    }


def test_future_builder_never_emits_legacy_camera_or_texture_labels() -> None:
    payload = _canonical_camera_descriptor().to_dict()
    assert "legacy_space_label" not in payload
    assert payload["space_id"] not in {"camera", "texture"}
    assert payload["profile_id"] not in {"camera", "texture"}
