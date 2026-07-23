from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

import fisheye.shared.keypoint_coordinate_publication as keypoint_publication_module
import fisheye.shared.subject_mask_coordinate_publication as publication_module
from fisheye.shared.coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    canonical_coordinate_descriptor_v2_digest,
)
from fisheye.shared.coordinate_record import coordinate_record_sha256
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_ATTR,
    DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
    DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    DirectedTransformV2Error,
    load_bound_directed_transform_v2,
)
from fisheye.shared.pixel_frame_authority import (
    CROP_PLACEMENT_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    PixelFrameAuthorityError,
    load_crop_placement_ownership,
)
from fisheye.shared.subject_mask_coordinate_publication import (
    SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR,
    SUBJECT_MASK_COMPONENT_LABELS_ATTR,
    SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR,
    SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
    SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
    SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
    SubjectMaskCoordinatePublicationError,
    _activate_validated_subject_mask_coordinate_surfaces,
    _load_completed_ineligible_subject_mask_coordinate_surfaces,
    capture_subject_mask_coordinate_publication_checkpoint,
    load_persisted_subject_mask_coordinate_surfaces,
    load_persisted_subject_mask_crop_source,
    prepare_subject_mask_coordinate_context,
    publish_subject_mask_coordinate_surfaces,
    rollback_subject_mask_coordinate_publication,
    selected_subject_mask_crop_values,
)
from fisheye.shared.model_input_transform import ModelInputTransform, resolve_model_input_transform
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    TransformAuthorityError,
    load_bound_transform_authority,
)
from tests.publication_fixture_clone import sealed_fixture_copy_memo
from tests.unit.fisheye.test_keypoint_coordinate_publication import (
    _MutableGroup,
    _fixture as keypoint_crop_fixture,
)


LABELS = ("subject_body", "eyes_union", "swim_bladder")
MODEL_TRANSFORM = resolve_model_input_transform((40, 40), mode="identity")
MODEL_ARTIFACT = {
    "role": "subject_mask_unet_checkpoint",
    "path": "/models/subject-mask.pt",
    "fingerprint_scheme": "content_v1",
    "sha256": "a" * 64,
    "size_bytes": 123,
    "mtime_ns": 456,
    "source": "computed",
}


def _owner(run: Any) -> str:
    return str(run.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR])


def _selector_snapshot(parent: Any) -> dict[str, tuple[bool, Any]]:
    return {
        name: (name in parent.attrs, copy.deepcopy(parent.attrs.get(name)))
        for name in (
            "latest",
            "latest_complete",
            "latest_pending",
            "authoritative_run",
            "authoritative_run_provenance",
            SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
            SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
            SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        )
    }


def _prepare_context(root: Any, **overrides: Any):
    run = root["subject_mask_runs/s1"]
    arguments = {
        "expected_publication_owner": _owner(run),
        "crop_path": "crop_runs/c1",
        "mask_labels": LABELS,
        "model_input_transform": MODEL_TRANSFORM,
        "model_artifact": MODEL_ARTIFACT,
        "mask_probability_threshold": 0.5,
    }
    arguments.update(overrides)
    return prepare_subject_mask_coordinate_context(
        root,
        "subject_mask_runs/s1",
        **arguments,
    )


def _publish(root: Any):
    run = root["subject_mask_runs/s1"]
    return publish_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=_owner(run),
    )


def _set_consistent_foreground(run: Any) -> None:
    probabilities = run["mask_probs_roi"].data
    probabilities[0, 0, 2:4, 5:8] = np.uint8(200)
    probabilities[0, 0, 2, 5] = np.uint8(255)
    run["masks_roi"].data[0, 0, 2:4, 5:8] = np.uint8(1)
    run["metrics"]["prob_max"].data[0, 0] = np.float32(1.0)
    run["metrics"]["mask_present"].data[0, 0] = True
    run["metrics"]["area_px"].data[0, 0] = np.float32(6.0)
    run["metrics"]["centroid_xy"].data[0, 0] = np.asarray(
        [6.0, 2.5],
        dtype=np.float32,
    )
    run["metrics"]["centroid_valid"].data[0, 0] = True
    run["metrics"]["bbox_xyxy"].data[0, 0] = np.asarray(
        [5.0, 2.0, 8.0, 4.0],
        dtype=np.float32,
    )
    run["metrics"]["bbox_valid"].data[0, 0] = True


def _surface(run: Any, path: str) -> Any:
    node = run
    for part in path.split("/"):
        node = node[part]
    return node


def _build_subject_fixture_with_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_identity: bool = True,
    roi_shape: tuple[int, int] = (40, 40),
) -> tuple[Any, _MutableGroup, _MutableGroup, Any]:
    root, _keypoints, _crop, _roi_images = keypoint_crop_fixture(monkeypatch)
    source = load_persisted_subject_mask_crop_source(root, "crop_runs/c1")
    _MutableGroup(
        path="analysis",
        root=root,
        token=root._coordinate_archive_token,
    )
    parent = _MutableGroup(
        path="subject_mask_runs",
        root=root,
        token=root._coordinate_archive_token,
    )
    run = parent.create_group("s1")
    run.attrs["stage_selector_eligible"] = False
    run.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR] = uuid4().hex
    run.attrs["mask_labels"] = list(LABELS)
    run.attrs["model_input_transform"] = MODEL_TRANSFORM.to_attrs()
    run.attrs["mask_probability_threshold"] = 0.5
    run.attrs["output_semantics"] = "multilabel"
    run.attrs["overlap_policy"] = "independent_sigmoid"
    run.attrs["probability_semantics"] = "sigmoid_multilabel_logits"
    run.attrs["probabilities_dtype"] = "uint8"
    run.attrs["probabilities_encoding"] = "linear_uint8_0_255"
    run.attrs["masks_roi_materialized"] = True
    run.attrs["binary_masks_materialized"] = True
    run.attrs["binary_masks_source"] = (
        "threshold(mask_probs_roi, threshold=0.5)"
    )
    run.attrs["bbox_xyxy_convention"] = "pixel_edge_half_open"
    run.attrs["bbox_xyxy_derivation"] = (
        "foreground_half_open_pixel_edges_xyxy_v1"
    )
    run.attrs["source_checkpoint"] = MODEL_ARTIFACT["path"]
    run.attrs["subject_mask_model_artifact"] = copy.deepcopy(MODEL_ARTIFACT)
    mark_run_started(run, run_name="s1", stage="subject_masks")

    selected = selected_subject_mask_crop_values(
        source,
        np.asarray([1, 0], dtype="<i8"),
    )
    run.create_array(
        "source_crop_row_ids",
        data=selected["source_crop_row_ids"],
        chunks=(2,),
    )
    if include_identity:
        run.create_array("instance_key", data=selected["instance_key"], chunks=(2,))
    run.create_array(
        "source_acquisition_frame_index",
        data=selected["source_acquisition_frame_index"],
        chunks=(2,),
    )
    run.create_array(
        "source_crop_xywh",
        data=selected["source_crop_xywh"],
        chunks=(2, 4),
    )

    n, c = 2, len(LABELS)
    h, w = roi_shape
    run.create_array(
        "mask_probs_roi",
        data=np.zeros((n, c, h, w), dtype=np.uint8),
        chunks=(1, 1, h, w),
    )
    run.create_array(
        "masks_roi",
        data=np.zeros((n, c, h, w), dtype=np.uint8),
        chunks=(1, 1, h, w),
    )
    metrics = run.create_group("metrics")
    metrics.create_array(
        "centroid_xy",
        data=np.zeros((n, c, 2), dtype=np.float32),
        chunks=(2, c, 2),
    )
    metrics.create_array(
        "bbox_xyxy",
        data=np.zeros((n, c, 4), dtype=np.float32),
        chunks=(2, c, 4),
    )
    run.create_array(
        "available_channels",
        data=np.ones((c,), dtype=bool),
        chunks=(c,),
    )
    metrics.create_array(
        "prob_max",
        data=np.zeros((n, c), dtype=np.float32),
        chunks=(n, c),
    )
    metrics.create_array(
        "mask_present",
        data=np.zeros((n, c), dtype=bool),
        chunks=(n, c),
    )
    metrics.create_array(
        "area_px",
        data=np.zeros((n, c), dtype=np.float32),
        chunks=(n, c),
    )
    metrics.create_array(
        "centroid_valid",
        data=np.zeros((n, c), dtype=bool),
        chunks=(n, c),
    )
    metrics.create_array(
        "bbox_valid",
        data=np.zeros((n, c), dtype=bool),
        chunks=(n, c),
    )
    return root, parent, run, source


_SUBJECT_FIXTURE_TEMPLATES: dict[
    tuple[bool, tuple[int, int], bool, str],
    tuple[Any, _MutableGroup, _MutableGroup, Any],
] = {}


def _copy_subject_fixture_template(
    monkeypatch: pytest.MonkeyPatch,
    template: tuple[Any, _MutableGroup, _MutableGroup, Any],
) -> tuple[Any, _MutableGroup, _MutableGroup, Any]:
    cloned = copy.deepcopy(template, sealed_fixture_copy_memo(template))
    root, _parent, _run, source = cloned
    monkeypatch.setattr(
        keypoint_publication_module,
        "load_persisted_crop_observation_geometry",
        lambda _root, _path: source.crop_geometry,
    )
    assert source._root is root
    return cloned


def _subject_fixture_with_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_identity: bool = True,
    roi_shape: tuple[int, int] = (40, 40),
    consistent_foreground: bool = False,
    prepared: bool = False,
    published: bool = False,
    fresh: bool = False,
) -> tuple[Any, _MutableGroup, _MutableGroup, Any]:
    if published:
        prepared = True
    if fresh:
        fixture = _build_subject_fixture_with_source(
            monkeypatch,
            include_identity=include_identity,
            roi_shape=roi_shape,
        )
        root, _parent, _run, _source = fixture
        if consistent_foreground:
            _set_consistent_foreground(_run)
        if prepared:
            _prepare_context(root)
        if published:
            _publish(root)
        return fixture

    state = "published" if published else "prepared" if prepared else "raw"
    key = (include_identity, roi_shape, consistent_foreground, state)
    template = _SUBJECT_FIXTURE_TEMPLATES.get(key)
    if template is None:
        if state == "raw":
            if consistent_foreground:
                plain_key = (include_identity, roi_shape, False, "raw")
                plain_template = _SUBJECT_FIXTURE_TEMPLATES.get(plain_key)
                if plain_template is None:
                    plain_template = _build_subject_fixture_with_source(
                        monkeypatch,
                        include_identity=include_identity,
                        roi_shape=roi_shape,
                    )
                    _SUBJECT_FIXTURE_TEMPLATES[plain_key] = plain_template
                template = _copy_subject_fixture_template(
                    monkeypatch,
                    plain_template,
                )
                _set_consistent_foreground(template[2])
            else:
                template = _build_subject_fixture_with_source(
                    monkeypatch,
                    include_identity=include_identity,
                    roi_shape=roi_shape,
                )
        else:
            raw_key = (
                include_identity,
                roi_shape,
                consistent_foreground,
                "raw",
            )
            raw_template = _SUBJECT_FIXTURE_TEMPLATES.get(raw_key)
            if raw_template is None:
                raw_template = _build_subject_fixture_with_source(
                    monkeypatch,
                    include_identity=include_identity,
                    roi_shape=roi_shape,
                )
                if consistent_foreground:
                    _set_consistent_foreground(raw_template[2])
                _SUBJECT_FIXTURE_TEMPLATES[raw_key] = raw_template
            template = _copy_subject_fixture_template(monkeypatch, raw_template)
            root, _parent, _run, _source = template
            _prepare_context(root)
            if state == "published":
                _publish(root)
        _SUBJECT_FIXTURE_TEMPLATES[key] = template
    return _copy_subject_fixture_template(monkeypatch, template)


def _subject_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_identity: bool = True,
    roi_shape: tuple[int, int] = (40, 40),
    consistent_foreground: bool = False,
    prepared: bool = False,
    published: bool = False,
    fresh: bool = False,
) -> tuple[Any, _MutableGroup, _MutableGroup]:
    root, parent, run, _source = _subject_fixture_with_source(
        monkeypatch,
        include_identity=include_identity,
        roi_shape=roi_shape,
        consistent_foreground=consistent_foreground,
        prepared=prepared,
        published=published,
        fresh=fresh,
    )
    return root, parent, run


def test_subject_mask_publication_binds_exact_crop_identity_labels_and_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, fresh=True)

    context = _prepare_context(root)
    assert context.labels == LABELS
    pending = _publish(root)
    assert pending.mask_probs_roi.descriptor.collection_axis is not None
    assert pending.mask_probs_roi.descriptor.collection_axis.cardinality == len(LABELS)
    assert pending.centroid_xy.descriptor.geometry_type == "point_xy"
    assert pending.bbox_xyxy.descriptor.geometry_type == "bbox_xyxy"
    assert pending.bbox_xyxy.descriptor.pixel_convention == "pixel_edge_half_open"
    assert pending.mask_probs_roi.descriptor.reference_extent.width == 40
    assert pending.mask_probs_roi.descriptor.reference_extent.height == 40

    selector_snapshot = _selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    assert "latest" not in parent.attrs
    complete = _load_completed_ineligible_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
    )
    _activate_validated_subject_mask_coordinate_surfaces(
        root,
        parent,
        complete,
        run_name="s1",
        publication_owner_token=run.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR],
        selector_snapshot=selector_snapshot,
    )
    loaded = load_persisted_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
    )

    assert parent.attrs["latest"] == "s1"
    assert parent.attrs["latest_complete"] == "s1"
    assert parent.attrs[SUBJECT_MASK_PUBLICATION_GENERATION_ATTR] == 1
    assert parent.attrs[SUBJECT_MASK_PUBLICATION_POLICY_ATTR].endswith(
        "selectors_then_eligibility_v1"
    )
    assert parent.attrs[SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR][
        "publication_owner"
    ] == _owner(run)
    assert run.attrs["stage_selector_eligible"] is True
    assert "source_crop_xywh_pixel_center" not in run
    assert loaded.context.labels == LABELS
    assert loaded.context.inference_authority.record_ref.endswith(
        f"@{SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR}"
    )
    assert loaded.interpretations["area_px"].record["units"] == "px^2"
    assert (
        loaded.interpretations["centroid_valid"].record[
            "geometry_relationship"
        ]["false_value_policy"]
        == "zero_xy_is_invalid_sentinel_not_coordinate"
    )
    for node in (
        run["mask_probs_roi"],
        run["masks_roi"],
        run["metrics"]["centroid_xy"],
        run["metrics"]["bbox_xyxy"],
    ):
        assert node.attrs[COORDINATE_DESCRIPTOR_ATTR]["collection_axis"][
            "label_authority"
        ]["record_ref"].endswith(f"@{SUBJECT_MASK_COMPONENT_LABELS_ATTR}")
        assert SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR in node.attrs
    for node in (
        run["available_channels"],
        run["metrics"]["prob_max"],
        run["metrics"]["mask_present"],
        run["metrics"]["area_px"],
        run["metrics"]["centroid_valid"],
        run["metrics"]["bbox_valid"],
    ):
        assert SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR in node.attrs


def test_canonical_success_scans_payload_once_at_publish_and_once_at_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, fresh=True)
    _prepare_context(root)
    original_validate = publication_module._validate_companion_metadata_and_values
    scans = 0

    def count_payload_scan(*args: Any, **kwargs: Any):
        nonlocal scans
        scans += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        publication_module,
        "_validate_companion_metadata_and_values",
        count_payload_scan,
    )
    published = _publish(root)
    selector_snapshot = _selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    _activate_validated_subject_mask_coordinate_surfaces(
        root,
        parent,
        published,
        run_name="s1",
        publication_owner_token=_owner(run),
        selector_snapshot=selector_snapshot,
    )

    assert scans == 2
    assert run.attrs["stage_selector_eligible"] is True


def test_activation_revalidates_payload_instead_of_trusting_published_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, fresh=True)
    _prepare_context(root)
    published = _publish(root)
    selector_snapshot = _selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    run["mask_probs_roi"].data[0, 0, 0, 0] = np.uint8(255)

    with pytest.raises(
        SubjectMaskCoordinatePublicationError,
        match="masks_roi must exactly equal",
    ):
        _activate_validated_subject_mask_coordinate_surfaces(
            root,
            parent,
            published,
            run_name="s1",
            publication_owner_token=_owner(run),
            selector_snapshot=selector_snapshot,
        )

    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs
    assert run.attrs["stage_selector_eligible"] is False


def test_activation_closes_completed_child_proof_before_parent_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, fresh=True)
    _prepare_context(root)
    published = _publish(root)
    selector_snapshot = _selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    original_finish = publication_module.finish_proof_verification
    original_acquire = publication_module._acquire_parent_publication_lease
    proof_closed = False

    def finish_before_parent_mutation() -> None:
        nonlocal proof_closed
        original_finish()
        proof_closed = True

    def acquire_after_proof_close(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert proof_closed is True
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(
        publication_module,
        "finish_proof_verification",
        finish_before_parent_mutation,
    )
    monkeypatch.setattr(
        publication_module,
        "_acquire_parent_publication_lease",
        acquire_after_proof_close,
    )

    _activate_validated_subject_mask_coordinate_surfaces(
        root,
        parent,
        published,
        run_name="s1",
        publication_owner_token=_owner(run),
        selector_snapshot=selector_snapshot,
    )

    assert proof_closed is True
    assert run.attrs["stage_selector_eligible"] is True


def test_subject_fixture_clones_do_not_mutate_each_other_or_the_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, first_parent, first_run, _first_source = (
        _subject_fixture_with_source(monkeypatch)
    )
    first_run["mask_probs_roi"].data[0, 0, 0, 0] = np.uint8(255)
    first_run.attrs["mask_probability_threshold"] = 0.75
    first_parent.attrs["latest"] = "mutated-clone"

    second_root, second_parent, second_run, second_source = (
        _subject_fixture_with_source(monkeypatch)
    )
    template_root, template_parent, template_run, _template_source = (
        _SUBJECT_FIXTURE_TEMPLATES[(True, (40, 40), False, "raw")]
    )

    assert first_root is not second_root
    assert first_root is not template_root
    assert second_root is not template_root
    assert "latest" not in second_parent.attrs
    assert "latest" not in template_parent.attrs
    assert second_run.attrs["mask_probability_threshold"] == 0.5
    assert template_run.attrs["mask_probability_threshold"] == 0.5
    assert int(second_run["mask_probs_roi"].data[0, 0, 0, 0]) == 0
    assert int(template_run["mask_probs_roi"].data[0, 0, 0, 0]) == 0
    assert not np.shares_memory(
        first_run["mask_probs_roi"].data,
        second_run["mask_probs_roi"].data,
    )
    assert second_source._root is second_root
    reloaded_source = load_persisted_subject_mask_crop_source(
        second_root,
        "crop_runs/c1",
    )
    assert reloaded_source.crop_geometry is second_source.crop_geometry


def test_subject_mask_placement_records_coexist_and_fail_closed_on_cross_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    context = _prepare_context(root)
    placement = run["source_crop_xywh"]
    expected_attrs = {
        CROP_PLACEMENT_OWNERSHIP_ATTR,
        f"{CROP_PLACEMENT_OWNERSHIP_ATTR}_sha256",
        CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
        f"{CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR}_sha256",
        TRANSFORM_AUTHORITY_ATTR,
        f"{TRANSFORM_AUTHORITY_ATTR}_sha256",
        TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
        f"{TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR}_sha256",
        DIRECTED_TRANSFORM_V2_ATTR,
        f"{DIRECTED_TRANSFORM_V2_ATTR}_sha256",
        DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
        f"{DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR}_sha256",
        CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
        f"{CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR}_sha256",
        TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
        f"{TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR}_sha256",
        DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
        f"{DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR}_sha256",
    }
    assert expected_attrs.issubset(placement.attrs)
    assert "source_crop_xywh_pixel_center" not in run

    continuous_camera = context.continuous_chain.source_camera_frame_authority
    with pytest.raises(PixelFrameAuthorityError, match="pixel_center"):
        load_crop_placement_ownership(
            placement,
            row_identity=context.row_identity,
            source_camera_frame=continuous_camera,
            attr_name=CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
        )
    with pytest.raises(PixelFrameAuthorityError, match="Unsupported"):
        load_crop_placement_ownership(
            placement,
            row_identity=context.row_identity,
            source_camera_frame=continuous_camera,
            attr_name="crop_placement_ownership_alias",
        )
    with pytest.raises(TransformAuthorityError, match="pixel-center"):
        load_bound_transform_authority(
            placement,
            payload_node=placement,
            source_frame=context.continuous_frame,
            target_frame=continuous_camera,
            row_identity=context.row_identity,
            attr_name=TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
        )
    pixel_link = context.pixel_center_chain.links[0]
    with pytest.raises(DirectedTransformV2Error, match="pixel-center"):
        load_bound_directed_transform_v2(
            placement,
            authority=pixel_link.authority,
            source_frame=context.continuous_frame,
            target_frame=continuous_camera,
            row_identity=context.row_identity,
            attr_name=DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
        )


def test_default_crop_publication_reloads_without_mutating_existing_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, _run = _subject_fixture(monkeypatch)
    source = load_persisted_subject_mask_crop_source(root, "crop_runs/c1")
    before = copy.deepcopy(dict(source._placement_node.attrs))

    reloaded = load_persisted_subject_mask_crop_source(root, "crop_runs/c1")

    assert dict(reloaded._placement_node.attrs) == before
    assert reloaded.crop_placement_ownership.record_ref.endswith(
        f"@{CROP_PLACEMENT_OWNERSHIP_ATTR}"
    )


@pytest.mark.parametrize(
    "attr_name",
    (
        CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
        TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
        DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
    ),
)
def test_subject_mask_pixel_center_chain_rejects_record_digest_tampering(
    monkeypatch: pytest.MonkeyPatch,
    attr_name: str,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, prepared=True)
    run["source_crop_xywh"].attrs[f"{attr_name}_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        publication_module._load_subject_mask_coordinate_context(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


def test_subject_mask_preflight_rejects_missing_instance_identity_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, include_identity=False)
    attrs_before = copy.deepcopy(dict(run.attrs))

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="instance_key"):
        _prepare_context(root)

    assert dict(run.attrs) == attrs_before
    assert "coordinate_frames" not in run


def test_subject_mask_child_rollback_preserves_concurrently_replaced_shared_camera_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    source = load_persisted_subject_mask_crop_source(root, "crop_runs/c1")
    camera_id = (
        source.crop_geometry.source_geometry.frame_evidence.acquisition_frame.record.camera_id
    )
    shared_path = (
        f"analysis/coordinate_frames/source_camera/{camera_id}/pixel_center"
    )
    shared_parent_path = shared_path.rsplit("/", 1)[0]
    original_stamp = publication_module.stamp_crop_placement_ownership
    replacement_holder: dict[str, Any] = {}

    def replace_shared_then_interrupt(*args: Any, **kwargs: Any):
        old_shared = root[shared_path]
        replacement = _MutableGroup(
            path=shared_path,
            root=root,
            token=root._coordinate_archive_token,
        )
        replacement.attrs.update(copy.deepcopy(dict(old_shared.attrs)))
        replacement.attrs["concurrent_writer_marker"] = "must_survive"
        root[shared_parent_path].children["pixel_center"] = replacement
        replacement_holder["node"] = replacement
        monkeypatch.setattr(
            publication_module,
            "stamp_crop_placement_ownership",
            original_stamp,
        )
        raise KeyboardInterrupt("synthetic concurrent shared-node replacement")

    monkeypatch.setattr(
        publication_module,
        "stamp_crop_placement_ownership",
        replace_shared_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt, match="shared-node replacement"):
        _prepare_context(root)

    replacement = replacement_holder["node"]
    assert root[shared_path] is replacement
    assert replacement.attrs["concurrent_writer_marker"] == "must_survive"
    assert "coordinate_frames" not in run


def test_subject_mask_publication_rejects_wrong_roi_reference_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, roi_shape=(39, 40))
    _prepare_context(root)

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="reference dimensions"):
        _publish(root)
    assert COORDINATE_DESCRIPTOR_ATTR not in run["mask_probs_roi"].attrs


def test_subject_mask_loader_rejects_ordered_label_authority_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, published=True)

    label_record = copy.deepcopy(run.attrs[SUBJECT_MASK_COMPONENT_LABELS_ATTR])
    label_record["labels"] = ["eyes_union", "subject_body", "swim_bladder"]
    run.attrs[SUBJECT_MASK_COMPONENT_LABELS_ATTR] = label_record

    with pytest.raises(ValueError):
        _publish(root)


def test_subject_mask_loader_rejects_equal_cardinality_descriptor_authority_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, published=True)

    node = run["mask_probs_roi"]
    payload = copy.deepcopy(node.attrs[COORDINATE_DESCRIPTOR_ATTR])
    payload["collection_axis"]["label_authority"] = {
        "record_ref": "/subject_mask_runs/s1@alternate_component_labels",
        "record_sha256": "f" * 64,
    }
    for lineage in payload["lineage_refs"]:
        if lineage["record_ref"].endswith(
            f"@{SUBJECT_MASK_COMPONENT_LABELS_ATTR}"
        ):
            lineage.update(payload["collection_axis"]["label_authority"])
    node.attrs[COORDINATE_DESCRIPTOR_ATTR] = payload
    node.attrs[f"{COORDINATE_DESCRIPTOR_ATTR}_sha256"] = (
        canonical_coordinate_descriptor_v2_digest(payload)
    )

    with pytest.raises(ValueError):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


def test_subject_mask_publication_checkpoint_rolls_back_baseexception_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, prepared=True)
    checkpoint = capture_subject_mask_coordinate_publication_checkpoint(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=_owner(run),
    )
    _publish(root)
    rollback_subject_mask_coordinate_publication(checkpoint)

    assert "coordinate_contract" not in run.attrs
    assert COORDINATE_DESCRIPTOR_ATTR not in run["mask_probs_roi"].attrs


def test_subject_mask_publication_operations_require_the_exact_expected_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    wrong_owner = "f" * 32
    assert wrong_owner != _owner(run)

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="another publication owner"):
        _prepare_context(root, expected_publication_owner=wrong_owner)
    _prepare_context(root)
    with pytest.raises(SubjectMaskCoordinatePublicationError, match="another publication owner"):
        capture_subject_mask_coordinate_publication_checkpoint(
            root,
            "subject_mask_runs/s1",
            expected_publication_owner=wrong_owner,
        )
    with pytest.raises(SubjectMaskCoordinatePublicationError, match="another publication owner"):
        publish_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            expected_publication_owner=wrong_owner,
        )


def test_subject_mask_checkpoint_refuses_replaced_child_and_leaves_stale_handle_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, prepared=True)
    checkpoint = capture_subject_mask_coordinate_publication_checkpoint(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=_owner(run),
    )
    _publish(root)
    replacement = _MutableGroup(
        path="subject_mask_runs/s1",
        root=root,
        token=root._coordinate_archive_token,
    )
    replacement.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR] = "e" * 32
    parent.children["s1"] = replacement

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="another publication owner"):
        rollback_subject_mask_coordinate_publication(checkpoint)
    assert run.attrs["coordinate_contract"] == "canonical_v2"
    assert "coordinate_contract" not in replacement.attrs


def test_subject_mask_checkpoint_refuses_eligible_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, prepared=True)
    checkpoint = capture_subject_mask_coordinate_publication_checkpoint(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=_owner(run),
    )
    _publish(root)
    mark_run_complete(run, parent_group=None, run_name="s1")
    run.attrs["stage_selector_eligible"] = True

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="selector-ineligible"):
        rollback_subject_mask_coordinate_publication(checkpoint)
    assert run.attrs["coordinate_contract"] == "canonical_v2"


@pytest.mark.parametrize(
    "transform",
    (
        ModelInputTransform(
            name="identity",
            native_height=39,
            native_width=40,
            model_height=39,
            model_width=40,
        ),
        ModelInputTransform(
            name="pad_to_size",
            native_height=40,
            native_width=40,
            model_height=39,
            model_width=40,
        ),
        ModelInputTransform(
            name="identity",
            native_height=40,
            native_width=40,
            model_height=40,
            model_width=40,
            pad_top=1,
        ),
    ),
)
def test_subject_mask_preflight_rejects_wrong_model_input_dimensions_or_padding(
    monkeypatch: pytest.MonkeyPatch,
    transform: ModelInputTransform,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    run.attrs["model_input_transform"] = transform.to_attrs()

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="model-input"):
        _prepare_context(root, model_input_transform=transform)


@pytest.mark.parametrize(
    ("attr_name", "replacement"),
    (
        ("mask_probability_threshold", 0.7),
        (
            "subject_mask_model_artifact",
            {**MODEL_ARTIFACT, "sha256": "b" * 64},
        ),
        ("source_checkpoint", "/models/replaced.pt"),
    ),
)
def test_subject_mask_loader_rejects_live_inference_evidence_tampering(
    monkeypatch: pytest.MonkeyPatch,
    attr_name: str,
    replacement: Any,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, prepared=True)
    run.attrs[attr_name] = replacement

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="inference authority"):
        publication_module._load_subject_mask_coordinate_context(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


def test_subject_mask_loader_rejects_structurally_invalid_transform_record_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, prepared=True)
    record = copy.deepcopy(run.attrs[SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR])
    record["model_input_transform"]["pad_top"] = 1
    run.attrs[SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR] = record
    run.attrs[f"{SUBJECT_MASK_INFERENCE_AUTHORITY_ATTR}_sha256"] = (
        coordinate_record_sha256(record)
    )
    run.attrs["model_input_transform"] = copy.deepcopy(
        record["model_input_transform"]
    )

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="padding"):
        publication_module._load_subject_mask_coordinate_context(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


def test_subject_mask_preflight_rejects_legacy_row_alias_even_when_contradictory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    run.create_array(
        "frame_indices",
        data=np.asarray([999, 998], dtype="<i8"),
        chunks=(2,),
    )

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="legacy row aliases"):
        _prepare_context(root)


def test_subject_mask_preflight_requires_detection_source_to_be_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    run.attrs["detection_source"] = "detect_runs/ambiguous"

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="omit detection_source"):
        _prepare_context(root)


@pytest.mark.parametrize(
    "missing_path",
    (
        "available_channels",
        "metrics/prob_max",
        "metrics/mask_present",
        "metrics/area_px",
        "metrics/centroid_valid",
        "metrics/bbox_valid",
    ),
)
def test_subject_mask_publication_requires_every_interpretation_companion(
    monkeypatch: pytest.MonkeyPatch,
    missing_path: str,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, prepared=True)
    if "/" in missing_path:
        group_name, array_name = missing_path.split("/", 1)
        run[group_name].children.pop(array_name)
    else:
        run.children.pop(missing_path)

    with pytest.raises(SubjectMaskCoordinatePublicationError):
        _publish(root)


def test_subject_mask_publication_supports_explicitly_omitted_binary_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    run.children.pop("masks_roi")
    run.attrs["masks_roi_materialized"] = False
    run.attrs["binary_masks_materialized"] = False
    run.attrs["binary_masks_source"] = "not_materialized"
    _prepare_context(root)

    pending = _publish(root)

    assert pending.masks_roi is None
    assert pending.inventory.record["optional_geometry"] == {"masks_roi": False}


def test_subject_mask_publication_rejects_binary_cache_presence_attr_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    run.attrs["masks_roi_materialized"] = False
    _prepare_context(root)

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="physically present"):
        _publish(root)


def test_subject_mask_loader_rejects_same_shape_validity_record_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, published=True)
    centroid_valid = run["metrics"]["centroid_valid"]
    bbox_valid = run["metrics"]["bbox_valid"]
    centroid_attrs = copy.deepcopy(dict(centroid_valid.attrs))
    bbox_attrs = copy.deepcopy(dict(bbox_valid.attrs))
    centroid_valid.attrs.clear()
    centroid_valid.attrs.update(bbox_attrs)
    bbox_valid.attrs.clear()
    bbox_valid.attrs.update(centroid_attrs)

    with pytest.raises(ValueError):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


def test_subject_mask_validity_makes_zero_sentinels_unambiguous_and_tamper_evident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, published=True)
    assert np.all(np.asarray(run["metrics"]["centroid_xy"][:]) == 0.0)
    assert not np.any(np.asarray(run["metrics"]["centroid_valid"][:]))

    run["metrics"]["centroid_valid"].data[0, 0] = True
    with pytest.raises(SubjectMaskCoordinatePublicationError, match="area_px>0"):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


@pytest.mark.parametrize("surface_name", ("mask_probs_roi", "centroid_xy"))
def test_subject_mask_loader_rejects_same_shape_value_mutation_even_when_derivation_still_matches(
    monkeypatch: pytest.MonkeyPatch,
    surface_name: str,
) -> None:
    root, _parent, run = _subject_fixture(
        monkeypatch,
        consistent_foreground=True,
        published=True,
    )

    if surface_name == "mask_probs_roi":
        # The thresholded mask, area, centroid, bbox, and prob_max are unchanged.
        run["mask_probs_roi"].data[0, 0, 2, 6] = np.uint8(201)
    else:
        # This remains inside the explicitly permitted float32 derivation tolerance.
        centroid = run["metrics"]["centroid_xy"].data
        centroid[0, 0, 0] = np.nextafter(
            centroid[0, 0, 0],
            np.float32(np.inf),
            dtype=np.float32,
        )

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="interpretation"):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


def test_subject_mask_loader_rejects_same_shape_probability_mask_role_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch, published=True)
    probabilities = run["mask_probs_roi"]
    masks = run["masks_roi"]
    probability_attrs = copy.deepcopy(dict(probabilities.attrs))
    mask_attrs = copy.deepcopy(dict(masks.attrs))
    probabilities.attrs.clear()
    probabilities.attrs.update(mask_attrs)
    masks.attrs.clear()
    masks.attrs.update(probability_attrs)

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="interpretation"):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


@pytest.mark.parametrize(
    ("path", "index", "replacement", "message"),
    (
        ("masks_roi", (0, 0, 2, 5), 0, "thresholded"),
        ("metrics/prob_max", (0, 0), 0.5, "prob_max"),
        ("metrics/area_px", (0, 0), 5.0, "area_px"),
        ("metrics/centroid_xy", (0, 0, 0), 6.5, "centroids"),
        ("metrics/bbox_xyxy", (0, 0, 0), 6.0, "bbox_xyxy"),
        ("metrics/centroid_valid", (0, 0), False, "area_px>0"),
    ),
)
def test_subject_mask_loader_recomputes_and_rejects_inconsistent_derivations(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    index: tuple[int, ...],
    replacement: Any,
    message: str,
) -> None:
    root, _parent, run = _subject_fixture(
        monkeypatch,
        consistent_foreground=True,
        published=True,
    )
    _surface(run, path).data[index] = replacement

    with pytest.raises(SubjectMaskCoordinatePublicationError, match=message):
        publication_module._load_subject_mask_coordinate_surfaces(
            root,
            "subject_mask_runs/s1",
            require_complete=False,
            expected_selector_eligible=False,
        )


def test_subject_mask_publication_rejects_nonfinite_float16_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run = _subject_fixture(monkeypatch)
    probabilities = np.zeros((2, len(LABELS), 40, 40), dtype=np.float16)
    probabilities[0, 0, 0, 0] = np.float16(np.nan)
    run.create_array(
        "mask_probs_roi",
        data=probabilities,
        chunks=(1, 1, 40, 40),
        overwrite=True,
    )
    run.attrs["probabilities_dtype"] = "float16"
    run.attrs["probabilities_encoding"] = "unit_float"
    _prepare_context(root)

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="finite values"):
        _publish(root)


@pytest.mark.parametrize(
    ("selector", "replacement"),
    (
        ("latest", "concurrent_latest"),
        ("authoritative_run", "concurrent_approved"),
        (
            "authoritative_run_provenance",
            {"approved_by": "concurrent-reviewer"},
        ),
        ("latest_pending", "another_attempt"),
    ),
)
def test_subject_mask_activation_rejects_concurrent_selector_mutation(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    replacement: Any,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, published=True)
    selector_snapshot = _selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    complete = _load_completed_ineligible_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=run.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR],
    )
    parent.attrs[selector] = replacement

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="selector"):
        _activate_validated_subject_mask_coordinate_surfaces(
            root,
            parent,
            complete,
            run_name="s1",
            publication_owner_token=run.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR],
            selector_snapshot=selector_snapshot,
        )
    assert run.attrs["stage_selector_eligible"] is False


@pytest.mark.parametrize(
    ("attr_name", "replacement"),
    (
        (SUBJECT_MASK_PUBLICATION_GENERATION_ATTR, 7),
        (SUBJECT_MASK_PUBLICATION_POLICY_ATTR, "unsupported_policy"),
        (
            SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
            {"publication_owner": "e" * 32},
        ),
    ),
)
def test_subject_mask_activation_rejects_concurrent_publication_epoch_mutation(
    monkeypatch: pytest.MonkeyPatch,
    attr_name: str,
    replacement: Any,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, published=True)
    selector_snapshot = _selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    complete = _load_completed_ineligible_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=_owner(run),
    )
    parent.attrs[attr_name] = replacement

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="publication state"):
        _activate_validated_subject_mask_coordinate_surfaces(
            root,
            parent,
            complete,
            run_name="s1",
            publication_owner_token=_owner(run),
            selector_snapshot=selector_snapshot,
        )
    assert run.attrs["stage_selector_eligible"] is False


def test_subject_mask_activation_rechecks_parent_lease_before_selector_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, published=True)
    selector_snapshot = _selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    complete = _load_completed_ineligible_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=_owner(run),
    )
    acquire = publication_module._acquire_parent_publication_lease

    def replace_after_acquire(*args: Any, **kwargs: Any) -> dict[str, Any]:
        lease = acquire(*args, **kwargs)
        parent.attrs[SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR] = {
            **lease,
            "publication_owner": "e" * 32,
        }
        return lease

    monkeypatch.setattr(
        publication_module,
        "_acquire_parent_publication_lease",
        replace_after_acquire,
    )
    with pytest.raises(SubjectMaskCoordinatePublicationError, match="lease was replaced"):
        _activate_validated_subject_mask_coordinate_surfaces(
            root,
            parent,
            complete,
            run_name="s1",
            publication_owner_token=_owner(run),
            selector_snapshot=selector_snapshot,
        )
    assert "latest_complete" not in parent.attrs
    assert run.attrs["stage_selector_eligible"] is False


def test_subject_mask_activation_rejects_publication_owner_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run = _subject_fixture(monkeypatch, published=True)
    selector_snapshot = _selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    complete = _load_completed_ineligible_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=run.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR],
    )
    original_owner = run.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR]
    run.attrs[SUBJECT_MASK_PUBLICATION_OWNER_ATTR] = "e" * 32

    with pytest.raises(SubjectMaskCoordinatePublicationError, match="another publication owner"):
        _activate_validated_subject_mask_coordinate_surfaces(
            root,
            parent,
            complete,
            run_name="s1",
            publication_owner_token=original_owner,
            selector_snapshot=selector_snapshot,
        )
    assert run.attrs["stage_selector_eligible"] is False
