from __future__ import annotations

import copy
import json

import pytest

from fisheye.shared.coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    COORDINATE_DESCRIPTOR_SCHEMA_ID,
    CoordinateDescriptorError,
    CoordinateRecordRef,
    LegacySpaceContext,
    build_coordinate_descriptor,
    canonical_coordinate_descriptor_json,
    coordinate_descriptor_attrs,
    coordinate_descriptor_digest,
    load_coordinate_descriptor_attrs,
    parse_coordinate_descriptor,
    resolve_legacy_space_id,
    stamp_coordinate_descriptor,
    validate_coordinate_descriptor,
)


def _camera_points_descriptor():
    return build_coordinate_descriptor(
        space_id="source_camera_image_px",
        geometry_type="points_xy",
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


def test_attrs_are_compact_and_detect_tampering() -> None:
    descriptor = _camera_points_descriptor()
    attrs = coordinate_descriptor_attrs(descriptor)

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
    assert stamp_coordinate_descriptor(node, descriptor) == descriptor
    assert node.attrs["existing"] is True
    assert load_coordinate_descriptor_attrs(node.attrs) == descriptor


def test_lineage_and_transform_provenance_remain_separate_references() -> None:
    descriptor = build_coordinate_descriptor(
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
            CoordinateRecordRef(ref="/coordinate_records/crop_1/source_selection"),
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
        {"ref": "/coordinate_records/crop_1/source_selection"}
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
        build_coordinate_descriptor(**kwargs)
    assert "physical_frame_required" in {
        issue.code for issue in exc_info.value.issues
    }

    descriptor = build_coordinate_descriptor(
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
    descriptor = build_coordinate_descriptor(
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
