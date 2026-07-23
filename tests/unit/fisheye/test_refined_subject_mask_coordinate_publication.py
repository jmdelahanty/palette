from __future__ import annotations

import copy
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

import fisheye.shared.keypoint_coordinate_publication as keypoint_publication_module
import fisheye.shared.refined_subject_mask_coordinate_publication as module
from fisheye.shared.array_measurement_descriptor import (
    ARRAY_MEASUREMENT_DESCRIPTOR_ATTR,
)
from fisheye.shared.coordinate_descriptor import COORDINATE_DESCRIPTOR_ATTR
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
    parse_directed_transform_v2,
)
from fisheye.shared.mask_geometry import batch_mask_spatial_metrics
from fisheye.shared.refined_subject_mask_coordinate_publication import (
    REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
    REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
    REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
    REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_ATTR,
    RefinedSubjectMaskCoordinatePublicationError,
    _activate_validated_refined_subject_mask_coordinate_surfaces,
    load_persisted_refined_subject_mask_coordinate_surfaces,
    prepare_refined_subject_mask_coordinate_context,
    publish_refined_subject_mask_coordinate_surfaces,
    require_bound_refined_subject_mask_coordinate_surfaces,
)
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started
from tests.publication_fixture_clone import sealed_fixture_copy_memo
from tests.unit.fisheye.test_keypoint_coordinate_publication import _MutableGroup
from tests.unit.fisheye.test_subject_mask_coordinate_publication import (
    _prepare_context as _prepare_raw_context,
    _publish as _publish_raw,
    _selector_snapshot as _raw_selector_snapshot,
    _set_consistent_foreground,
    _subject_fixture_with_source,
)


def test_shared_mask_geometry_uses_half_open_pixel_edge_bboxes() -> None:
    masks = np.zeros((2, 6, 8), dtype=np.uint8)
    masks[0, 2:5, 3:7] = 1

    metrics = batch_mask_spatial_metrics(masks)

    np.testing.assert_array_equal(
        metrics["bbox_xyxy"],
        np.asarray([[3.0, 2.0, 7.0, 5.0], [0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(metrics["bbox_valid"], np.asarray([True, False]))


def _snapshot(parent: Any) -> dict[str, tuple[bool, Any]]:
    return {
        name: (name in parent.attrs, copy.deepcopy(parent.attrs.get(name)))
        for name in (
            "latest",
            "latest_complete",
            "latest_pending",
            "authoritative_run",
            "authoritative_run_provenance",
            REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
            REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        )
    }


def _activate_raw(root: Any, parent: Any, run: Any) -> None:
    run.attrs["provenance"] = {
        "stage": "subject_masks",
        "method": "unit_test_canonical_raw",
        "source": "crop_runs/c1",
    }
    _set_consistent_foreground(run)
    _prepare_raw_context(root)
    pending = _publish_raw(root)
    snapshot = _raw_selector_snapshot(parent)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    from fisheye.shared.subject_mask_coordinate_publication import (
        _activate_validated_subject_mask_coordinate_surfaces,
        _load_completed_ineligible_subject_mask_coordinate_surfaces,
    )

    fresh = _load_completed_ineligible_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
    )
    assert fresh.derivation.record_sha256 == pending.derivation.record_sha256
    _activate_validated_subject_mask_coordinate_surfaces(
        root,
        parent,
        fresh,
        run_name="s1",
        publication_owner_token=run.attrs["subject_mask_publication_owner"],
        selector_snapshot=snapshot,
    )


def _copy_array(target: Any, source: Any, name: str) -> None:
    values = np.asarray(source[name][:]).copy()
    target.create_array(
        name,
        data=values,
        chunks=tuple(int(value) for value in values.shape),
    )


def _add_component_contract_surfaces(run: Any, raw_run: Any) -> Any:
    run.attrs.update(
        {
            "derived_mask_caches_stale": False,
            "metrics_stale": False,
            "contours_stale": False,
        }
    )
    components = run.create_group("components")
    labels = tuple(str(value) for value in run.attrs["mask_labels"])
    raw_labels = tuple(str(value) for value in raw_run.attrs["mask_labels"])
    for component_index, component in enumerate(labels):
        source_component = component if component in raw_labels else "eyes_union"
        group = components.create_group(component)
        group.create_array(
            "area_px",
            data=np.asarray(run["metrics"]["area_px"][:, component_index]).copy(),
            chunks=(2,),
        )
        group.create_array(
            "mask_present",
            data=np.asarray(run["metrics"]["mask_present"][:, component_index]).copy(),
            chunks=(2,),
        )
        provenance = group.create_group("provenance")
        source_path = "subject_mask_runs/s1/mask_probs_roi"
        provenance.attrs.update(
            {
                "source_channels": [source_component],
                "source_surface_path": source_path,
                "source_surface_kind": "probability",
                "source_probability_path": source_path,
                "source_probability_encoding": raw_run.attrs["probabilities_encoding"],
                "source_probability_threshold": float(
                    raw_run.attrs["mask_probability_threshold"]
                ),
                "source_binary_derivation": "smart_finalize(mask_probs_roi)",
                "finalization_method": "smart_finalize_subject_masks_v1",
                "finalization_policy": {"fixture": "exact_source_selection_v1"},
            }
        )
    return components


def _build_refined_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_optional_geometry: bool = True,
    with_eye_relation: bool = False,
) -> tuple[Any, Any, Any, Any, dict[str, tuple[bool, Any]], Any]:
    root, raw_parent, raw_run, crop_source = _subject_fixture_with_source(monkeypatch)
    _activate_raw(root, raw_parent, raw_run)
    parent = _MutableGroup(
        path="refined_subject_masks_runs",
        root=root,
        token=root._coordinate_archive_token,
    )
    snapshot = _snapshot(parent)
    run = parent.create_group("r1")
    refined_labels = (
        ["subject_body", "eye_left", "eye_right", "swim_bladder"]
        if with_eye_relation
        else list(raw_run.attrs["mask_labels"])
    )
    run.attrs.update(
        {
            "stage_selector_eligible": False,
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR: uuid4().hex,
            "source_subject_mask_run": "s1",
            "mask_labels": refined_labels,
            "label_schema_id": "subject_v1_union",
            "method": "smart_finalize_subject_masks_v1",
            "refinement_semantics": "canonical_component_masks",
            "finalization_semantics": "smart_probability_to_refined_candidate",
            "bbox_xyxy_convention": "pixel_edge_half_open",
            "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
            "masks_roi_materialized": True,
            "provenance": {
                "stage": "refine_subject_masks",
                "method": "smart_finalize_subject_masks_v1",
                "inputs": {"source_subject_mask_run": "s1"},
            },
        }
    )
    mark_run_started(run, run_name="r1", stage="refine_subject_masks")
    parent.attrs["latest_pending"] = "r1"
    for name in (
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "source_crop_xywh",
    ):
        _copy_array(run, raw_run, name)
    if with_eye_relation:
        raw_masks = np.asarray(raw_run["masks_roi"][:])
        masks = np.zeros((raw_masks.shape[0], 4, *raw_masks.shape[2:]), dtype=np.uint8)
        masks[:, 0] = raw_masks[:, 0]
        masks[:, 3] = raw_masks[:, 2]
        masks[0, 1, 9:12, 9:12] = 1
        masks[0, 2, 13:16, 12:15] = 1
        run.create_array("masks_roi", data=masks, chunks=masks.shape)
        run.create_array(
            "available_channels",
            data=np.ones((4,), dtype=bool),
            chunks=(4,),
        )
    else:
        _copy_array(run, raw_run, "available_channels")
        _copy_array(run, raw_run, "masks_roi")
    metrics = run.create_group("metrics")
    if with_eye_relation:
        for name, values in module._derive_mask_metrics(np.asarray(run["masks_roi"][:])).items():
            metrics.create_array(name, data=values, chunks=values.shape)
    else:
        for name in (
            "mask_present",
            "area_px",
            "centroid_xy",
            "centroid_valid",
            "bbox_xyxy",
            "bbox_valid",
        ):
            _copy_array(metrics, raw_run["metrics"], name)
    components = _add_component_contract_surfaces(run, raw_run)
    primary_component = components[str(run.attrs["mask_labels"][0])]
    component_metrics = primary_component.create_group("metrics")
    component_metrics.attrs["schema_id"] = "refined_subject_component_mask_metrics_v1"
    component_metrics.create_array(
        "component_count",
        data=np.asarray([1, 0], dtype=np.int32),
        chunks=(2,),
    )
    finalization_metrics = primary_component.create_group("finalization_metrics")
    finalization_metrics.attrs["schema_id"] = (
        "refined_subject_component_finalization_metrics_v1"
    )
    finalization_metrics.create_array(
        "quality_code",
        data=np.asarray([0, 1], dtype=np.int16),
        chunks=(2,),
    )
    finalization_metrics.create_array(
        "quality_score",
        data=np.asarray([0.0, 1.0], dtype=np.float32),
        chunks=(2,),
    )

    if with_optional_geometry:
        component = components[str(run.attrs["mask_labels"][0])]
        geometry = component.create_group("geometry")
        ellipse = np.full((2, 5), np.nan, dtype=np.float32)
        geometry.create_array("ellipse_params", data=ellipse, chunks=(2, 5))
        geometry.create_array(
            "ellipse_success",
            data=np.zeros((2,), dtype=bool),
            chunks=(2,),
        )
        sampled = component.create_group("sampled_contours")
        sampled_points = np.full((2, 4, 2), np.nan, dtype=np.float32)
        sampled_points[0] = np.asarray(
            [[5.0, 2.0], [7.0, 2.0], [7.0, 3.0], [5.0, 3.0]],
            dtype=np.float32,
        )
        sampled.create_array("points_xy", data=sampled_points, chunks=(2, 4, 2))
        sampled.create_array(
            "valid",
            data=np.asarray([True, False], dtype=bool),
            chunks=(2,),
        )
        sampled.create_array(
            "source_point_count",
            data=np.asarray([4, 0], dtype=np.int32),
            chunks=(2,),
        )
        contours = component.create_group("contours")
        contours.attrs["points_placeholder_when_empty"] = False
        contours.create_array(
            "ptr",
            data=np.asarray([0, -1], dtype=np.int64),
            chunks=(2,),
        )
        contours.create_array(
            "len",
            data=np.asarray([4, 0], dtype=np.int32),
            chunks=(2,),
        )
        contours.create_array(
            "points_xy",
            data=sampled_points[0].copy(),
            chunks=(4, 2),
        )
    if with_eye_relation:
        for component_name, center in (
            ("eye_left", (10.0, 10.0)),
            ("eye_right", (13.0, 14.0)),
        ):
            geometry = components[component_name].create_group("geometry")
            ellipse = np.full((2, 5), np.nan, dtype=np.float32)
            ellipse[0] = np.asarray([*center, 3.0, 4.0, 0.0], dtype=np.float32)
            geometry.create_array("ellipse_params", data=ellipse, chunks=(2, 5))
            geometry.create_array(
                "ellipse_success",
                data=np.asarray([True, False], dtype=bool),
                chunks=(2,),
            )
        relation_metrics = (
            run.create_group("relations")
            .create_group("eye_pair")
            .create_group("metrics")
        )
        relation_metrics.attrs.update(
            {
                "relation_components": ["eye_left", "eye_right"],
                "relation_method": "ellipse_centroid_distance",
                "source_measurement": "unit_test_eye_geometry",
            }
        )
        relation_metrics.create_array(
            "separation_px",
            data=np.asarray([5.0, np.nan], dtype=np.float32),
            chunks=(2,),
        )
        relation_metrics.create_array(
            "separation_valid",
            data=np.asarray([True, False], dtype=bool),
            chunks=(2,),
        )
    return root, parent, run, raw_run, snapshot, crop_source


_REFINED_FIXTURE_TEMPLATES: dict[
    tuple[bool, bool, bool],
    tuple[Any, Any, Any, Any, dict[str, tuple[bool, Any]], Any],
] = {}


def _copy_refined_fixture_template(
    monkeypatch: pytest.MonkeyPatch,
    template: tuple[Any, Any, Any, Any, dict[str, tuple[bool, Any]], Any],
) -> tuple[Any, Any, Any, Any, dict[str, tuple[bool, Any]], Any]:
    cloned = copy.deepcopy(
        template,
        sealed_fixture_copy_memo(template),
    )
    root, _parent, _run, _raw_run, _snapshot, crop_source = cloned
    monkeypatch.setattr(
        keypoint_publication_module,
        "load_persisted_crop_observation_geometry",
        lambda _root, _path: crop_source.crop_geometry,
    )
    assert crop_source._root is root
    return cloned


def _clone_refined_fixture_template(
    monkeypatch: pytest.MonkeyPatch,
    template: tuple[Any, Any, Any, Any, dict[str, tuple[bool, Any]], Any],
) -> tuple[Any, Any, Any, Any, dict[str, tuple[bool, Any]]]:
    root, parent, run, raw_run, snapshot, _crop_source = (
        _copy_refined_fixture_template(monkeypatch, template)
    )
    return root, parent, run, raw_run, snapshot


def _refined_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_optional_geometry: bool = True,
    with_eye_relation: bool = False,
    published: bool = False,
    fresh: bool = False,
) -> tuple[Any, Any, Any, Any, dict[str, tuple[bool, Any]]]:
    if fresh:
        root, parent, run, raw_run, snapshot, _crop_source = _build_refined_fixture(
            monkeypatch,
            with_optional_geometry=with_optional_geometry,
            with_eye_relation=with_eye_relation,
        )
        if published:
            _publish_activate(root, parent, run, snapshot)
        return root, parent, run, raw_run, snapshot

    key = (with_optional_geometry, with_eye_relation, published)
    template = _REFINED_FIXTURE_TEMPLATES.get(key)
    if template is None:
        if published:
            unpublished_key = (with_optional_geometry, with_eye_relation, False)
            unpublished = _REFINED_FIXTURE_TEMPLATES.get(unpublished_key)
            if unpublished is None:
                unpublished = _build_refined_fixture(
                    monkeypatch,
                    with_optional_geometry=with_optional_geometry,
                    with_eye_relation=with_eye_relation,
                )
                _REFINED_FIXTURE_TEMPLATES[unpublished_key] = unpublished
            template = _copy_refined_fixture_template(monkeypatch, unpublished)
            root, parent, run, _raw_run, snapshot, _crop_source = template
            _publish_activate(root, parent, run, snapshot)
        else:
            template = _build_refined_fixture(
                monkeypatch,
                with_optional_geometry=with_optional_geometry,
                with_eye_relation=with_eye_relation,
            )
        _REFINED_FIXTURE_TEMPLATES[key] = template
    return _clone_refined_fixture_template(monkeypatch, template)


def _prepare(root: Any, run: Any):
    return prepare_refined_subject_mask_coordinate_context(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=run.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
        source_subject_mask_path="subject_mask_runs/s1",
        mask_labels=run.attrs["mask_labels"],
    )


def _add_minimal_refined_run(
    root: Any,
    parent: Any,
    raw_run: Any,
    *,
    run_name: str,
) -> tuple[Any, dict[str, tuple[bool, Any]]]:
    snapshot = _snapshot(parent)
    run = parent.create_group(run_name)
    run.attrs.update(
        {
            "stage_selector_eligible": False,
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR: uuid4().hex,
            "source_subject_mask_run": "s1",
            "mask_labels": list(raw_run.attrs["mask_labels"]),
            "label_schema_id": "subject_v1_union",
            "method": "smart_finalize_subject_masks_v1",
            "refinement_semantics": "canonical_component_masks",
            "finalization_semantics": "smart_probability_to_refined_candidate",
            "bbox_xyxy_convention": "pixel_edge_half_open",
            "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
            "masks_roi_materialized": True,
            "provenance": {
                "stage": "refine_subject_masks",
                "method": "smart_finalize_subject_masks_v1",
                "inputs": {"source_subject_mask_run": "s1"},
            },
        }
    )
    mark_run_started(run, run_name=run_name, stage="refine_subject_masks")
    for name in (
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "source_crop_xywh",
        "available_channels",
        "masks_roi",
    ):
        _copy_array(run, raw_run, name)
    metrics = run.create_group("metrics")
    for name in (
        "mask_present",
        "area_px",
        "centroid_xy",
        "centroid_valid",
        "bbox_xyxy",
        "bbox_valid",
    ):
        _copy_array(metrics, raw_run["metrics"], name)
    _add_component_contract_surfaces(run, raw_run)
    parent.attrs["latest_pending"] = run_name
    return run, snapshot


def _publish_activate(
    root: Any,
    parent: Any,
    run: Any,
    snapshot: dict[str, tuple[bool, Any]],
):
    _prepare(root, run)
    pending = publish_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=run.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
    )
    mark_run_complete(run, parent_group=None, run_name="r1")
    _activate_validated_refined_subject_mask_coordinate_surfaces(
        root,
        parent,
        pending,
        run_name="r1",
        publication_owner_token=run.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
        selector_snapshot=snapshot,
    )
    return load_persisted_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r1",
    )


def test_refined_fixture_clones_do_not_mutate_each_other_or_the_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root, first_parent, first_run, _first_raw, _first_snapshot = (
        _refined_fixture(monkeypatch)
    )
    first_run["masks_roi"].data[0, 0, 2, 5] = np.uint8(0)
    first_run.attrs["provenance"]["method"] = "mutated_clone"
    del first_parent.attrs["latest_pending"]

    second_root, second_parent, second_run, _second_raw, _second_snapshot = (
        _refined_fixture(monkeypatch)
    )
    template_root, template_parent, template_run, *_rest = (
        _REFINED_FIXTURE_TEMPLATES[(True, False, False)]
    )

    assert first_root is not second_root
    assert first_root is not template_root
    assert second_root is not template_root
    assert second_parent.attrs["latest_pending"] == "r1"
    assert template_parent.attrs["latest_pending"] == "r1"
    assert second_run.attrs["provenance"]["method"] == (
        "smart_finalize_subject_masks_v1"
    )
    assert template_run.attrs["provenance"]["method"] == (
        "smart_finalize_subject_masks_v1"
    )
    assert int(second_run["masks_roi"].data[0, 0, 2, 5]) == 1
    assert int(template_run["masks_roi"].data[0, 0, 2, 5]) == 1
    assert not np.shares_memory(
        first_run["masks_roi"].data,
        second_run["masks_roi"].data,
    )


def test_refined_publication_binds_dense_roi_geometry_and_exact_source_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run, _raw, snapshot = _refined_fixture(monkeypatch, fresh=True)

    loaded = _publish_activate(root, parent, run, snapshot)

    assert parent.attrs["latest"] == "r1"
    assert parent.attrs["latest_complete"] == "r1"
    assert run.attrs["stage_selector_eligible"] is True
    assert loaded.masks_roi.descriptor.space_id == "roi_local_px"
    assert loaded.masks_roi.descriptor.reference_extent.width == 40
    assert loaded.masks_roi.descriptor.reference_extent.height == 40
    assert loaded.bbox_xyxy.descriptor.pixel_convention == "pixel_edge_half_open"
    collection_axis = loaded.masks_roi.descriptor.collection_axis
    assert collection_axis is not None
    assert collection_axis.cardinality == len(run.attrs["mask_labels"])
    assert loaded.context.component_labels.record["role"] == "subject_component"
    assert loaded.context.context_record.record["roi_to_source_camera"]["direction"] == (
        "roi_local_px_to_source_camera_image_px"
    )
    placements = np.asarray(run["source_crop_xywh"][:])
    assert bool(np.any(placements[:, :2] != 0))
    assert "components/subject_body/geometry/ellipse_params" in loaded.descriptors
    assert "components/subject_body/sampled_contours/points_xy" in loaded.descriptors
    assert "relations/eye_pair/metrics/separation_px" not in loaded.descriptors
    assert "metrics/area_px" in loaded.measurements
    assert loaded.measurements["metrics/area_px"].record["semantic_kind"] == "area"
    assert loaded.measurements[
        "components/subject_body/metrics/component_count"
    ].record["semantic_kind"] == "count"
    assert loaded.measurements[
        "components/subject_body/finalization_metrics/quality_code"
    ].record["semantic_kind"] == "categorical"
    ragged = loaded.ragged_geometry[
        "components/subject_body/contours/points_xy"
    ]
    assert ragged.record["ragged_row_mapping"]["policy"].startswith("ptr_len")
    assert require_bound_refined_subject_mask_coordinate_surfaces(loaded).inventory.record_sha256 == (
        loaded.inventory.record_sha256
    )


def test_eye_separation_is_a_recomputed_measurement_not_a_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run, _raw, snapshot = _refined_fixture(
        monkeypatch,
        with_eye_relation=True,
        fresh=True,
    )

    loaded = _publish_activate(root, parent, run, snapshot)

    path = "relations/eye_pair/metrics/separation_px"
    assert path not in loaded.descriptors
    record = loaded.measurements[path].record
    assert record["quantity"] == "eye_pair_separation"
    assert record["units"] == "px"
    assert record["selected_collection_members"] == ["eye_left", "eye_right"]
    assert len(record["source_coordinate_descriptors"]) == 2

    run["relations"]["eye_pair"]["metrics"]["separation_px"].data[0] = np.float32(4.0)
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="exact selected eye ellipse centers",
    ):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )


def test_refined_activation_replaces_a_committed_lease_with_next_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, first, raw, first_snapshot = _refined_fixture(
        monkeypatch,
        fresh=True,
    )
    _publish_activate(root, parent, first, first_snapshot)
    first_lease = copy.deepcopy(
        parent.attrs[REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR]
    )
    second, second_snapshot = _add_minimal_refined_run(
        root,
        parent,
        raw,
        run_name="r2",
    )
    prepare_refined_subject_mask_coordinate_context(
        root,
        "refined_subject_masks_runs/r2",
        expected_publication_owner=second.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
        source_subject_mask_path="subject_mask_runs/s1",
        mask_labels=second.attrs["mask_labels"],
    )
    pending = publish_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r2",
        expected_publication_owner=second.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
    )
    mark_run_complete(second, parent_group=None, run_name="r2")
    _activate_validated_refined_subject_mask_coordinate_surfaces(
        root,
        parent,
        pending,
        run_name="r2",
        publication_owner_token=second.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
        selector_snapshot=second_snapshot,
    )

    assert parent.attrs[REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR] == 2
    assert parent.attrs["latest"] == "r2"
    assert parent.attrs["latest_complete"] == "r2"
    second_lease = parent.attrs[REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR]
    assert second_lease["base_generation"] == 1
    assert second_lease["next_generation"] == 2
    assert second_lease["publication_owner"] != first_lease["publication_owner"]
    assert second.attrs["stage_selector_eligible"] is True


@pytest.mark.parametrize("name", ("instance_key", "source_crop_row_ids"))
def test_refined_preflight_rejects_identity_or_source_row_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    run[name].data[0] += 1

    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="exact dtype-preserving copy",
    ):
        _prepare(root, run)

    assert "coordinate_frames" not in run


def test_refined_preflight_rejects_unequal_roi_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    original = np.asarray(run["masks_roi"][:])
    run.create_array(
        "masks_roi",
        data=original[..., :-1],
        chunks=(2, 3, 40, 39),
        overwrite=True,
    )

    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="extent differs",
    ):
        _prepare(root, run)


@pytest.mark.parametrize("tamper", ("values", "convention"))
def test_refined_publication_rejects_inclusive_or_undeclared_bbox(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    _prepare(root, run)
    if tamper == "values":
        run["metrics"]["bbox_xyxy"].data[0, 0, 2:] -= 1.0
    else:
        run.attrs["bbox_xyxy_convention"] = "pixel_center_inclusive"

    with pytest.raises(RefinedSubjectMaskCoordinatePublicationError):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )


@pytest.mark.parametrize(
    ("tamper", "match"),
    (
        ("source_path", "source surface leaves"),
        ("threshold", "probability encoding, threshold"),
    ),
)
def test_refined_preflight_rejects_inexact_component_source_selection(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    match: str,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    provenance = run["components"]["subject_body"]["provenance"]
    if tamper == "source_path":
        provenance.attrs["source_surface_path"] = "subject_mask_runs/other/mask_probs_roi"
    else:
        provenance.attrs["source_probability_threshold"] = 0.25

    with pytest.raises(RefinedSubjectMaskCoordinatePublicationError, match=match):
        _prepare(root, run)


def test_refined_publication_rejects_stale_flags_and_unknown_root_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    _prepare(root, run)
    run.attrs["metrics_stale"] = True
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="metrics_stale=False",
    ):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )
    run.attrs["metrics_stale"] = False
    run.create_array(
        "mystery_scalar",
        data=np.zeros((2,), dtype=np.float32),
        chunks=(2,),
    )
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="unsupported root array",
    ):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )


def test_refined_loader_rejects_descriptor_drop_or_payload_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot = _refined_fixture(
        monkeypatch,
        published=True,
    )
    descriptor = copy.deepcopy(run["masks_roi"].attrs[COORDINATE_DESCRIPTOR_ATTR])
    descriptor_digest = run["masks_roi"].attrs[f"{COORDINATE_DESCRIPTOR_ATTR}_sha256"]
    del run["masks_roi"].attrs[COORDINATE_DESCRIPTOR_ATTR]
    with pytest.raises(ValueError):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )
    run["masks_roi"].attrs[COORDINATE_DESCRIPTOR_ATTR] = descriptor
    run["masks_roi"].attrs[f"{COORDINATE_DESCRIPTOR_ATTR}_sha256"] = descriptor_digest
    run["masks_roi"].data[0, 0, 2, 5] = np.uint8(0)
    with pytest.raises(RefinedSubjectMaskCoordinatePublicationError):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )


def test_refined_loader_rejects_measurement_descriptor_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot = _refined_fixture(
        monkeypatch,
        published=True,
    )
    node = run["metrics"]["area_px"]
    del node.attrs[ARRAY_MEASUREMENT_DESCRIPTOR_ATTR]

    with pytest.raises(ValueError):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )


def test_refined_loader_rejects_wrong_transform_direction_with_fresh_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot = _refined_fixture(
        monkeypatch,
        published=True,
    )
    placement = run["source_crop_xywh"]
    record = copy.deepcopy(placement.attrs[DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR])
    record["from_space_id"], record["to_space_id"] = (
        record["to_space_id"],
        record["from_space_id"],
    )
    record["source"], record["target"] = record["target"], record["source"]
    placement.attrs[DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR] = record
    placement.attrs[f"{DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR}_sha256"] = (
        parse_directed_transform_v2(record).digest()
    )

    with pytest.raises(ValueError):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )


def test_refined_loader_rejects_stale_raw_or_refinement_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, raw, _snapshot = _refined_fixture(
        monkeypatch,
        published=True,
    )
    original_refinement_provenance = copy.deepcopy(run.attrs["provenance"])
    raw.attrs["provenance"] = {
        **raw.attrs["provenance"],
        "method": "tampered_raw_method",
    }
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="source authority",
    ):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )
    raw.attrs["provenance"]["method"] = "unit_test_canonical_raw"
    run.attrs["provenance"] = original_refinement_provenance
    run.attrs["run_provenance"] = {
        **run.attrs["run_provenance"],
        "command": "contradictory_completion_command",
    }
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="mechanically derived",
    ):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )
    run.attrs["provenance"] = {
        **run.attrs["provenance"],
        "method": "tampered_refinement",
    }
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="refinement authority",
    ):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )


def test_refined_publication_rolls_back_on_baseexception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    _prepare(root, run)
    monkeypatch.setattr(
        module,
        "stamp_bound_canonical_coordinate_descriptors",
        lambda _values: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )

    assert "coordinate_contract" not in run.attrs
    assert COORDINATE_DESCRIPTOR_ATTR not in run["masks_roi"].attrs
    assert REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_ATTR not in (
        run["components"]["subject_body"]["contours"]["points_xy"].attrs
    )


def test_refined_activation_interrupt_restores_owned_selectors_and_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run, _raw, snapshot = _refined_fixture(monkeypatch)
    _prepare(root, run)
    fresh = publish_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=run.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
    )
    mark_run_complete(run, parent_group=None, run_name="r1")

    class InterruptOnLatest(dict):
        interrupted = False

        def __setitem__(self, key: str, value: Any) -> None:
            if key == "latest" and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt()
            super().__setitem__(key, value)

    parent.attrs = InterruptOnLatest(parent.attrs)
    with pytest.raises(KeyboardInterrupt):
        _activate_validated_refined_subject_mask_coordinate_surfaces(
            root,
            parent,
            fresh,
            run_name="r1",
            publication_owner_token=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
            selector_snapshot=snapshot,
        )
    assert "latest" not in parent.attrs
    assert "latest_complete" not in parent.attrs
    assert "latest_pending" not in parent.attrs
    assert REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR not in parent.attrs
    assert run.attrs["stage_selector_eligible"] is False


def test_refined_late_activation_interrupt_restores_exact_preexisting_parent_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run, _raw, snapshot = _refined_fixture(monkeypatch)
    old_state = {
        "latest": "r0",
        "latest_complete": "r0",
        "latest_pending": "older_pending",
        "authoritative_run": "r0",
        "authoritative_run_provenance": {"approved_by": "unit-test"},
        REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR: 7,
        REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR: module._PUBLICATION_POLICY,
        REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR: {
            "schema_id": "palette.refined_subject_mask_publication_lease",
            "schema_version": 1,
            "policy": module._PUBLICATION_POLICY,
            "run_path": "refined_subject_masks_runs/r0",
            "publication_owner": "0" * 32,
            "base_generation": 6,
            "next_generation": 7,
        },
    }
    for name, value in old_state.items():
        snapshot[name] = (True, copy.deepcopy(value))
        if name != "latest_pending":
            parent.attrs[name] = copy.deepcopy(value)
    parent.attrs["latest_pending"] = "r1"
    _prepare(root, run)
    fresh = publish_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=run.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
    )
    mark_run_complete(run, parent_group=None, run_name="r1")

    class InterruptOnGeneration(dict):
        interrupted = False

        def __setitem__(self, key: str, value: Any) -> None:
            if (
                key == REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR
                and value == 8
                and not self.interrupted
            ):
                self.interrupted = True
                raise KeyboardInterrupt()
            super().__setitem__(key, value)

    parent.attrs = InterruptOnGeneration(parent.attrs)
    with pytest.raises(KeyboardInterrupt):
        _activate_validated_refined_subject_mask_coordinate_surfaces(
            root,
            parent,
            fresh,
            run_name="r1",
            publication_owner_token=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
            selector_snapshot=snapshot,
        )

    for name, value in old_state.items():
        assert parent.attrs[name] == value
    assert run.attrs["stage_selector_eligible"] is False


def test_refined_activation_prelease_rejects_generation_policy_lease_and_pending_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run, _raw, snapshot = _refined_fixture(monkeypatch)
    _prepare(root, run)
    pending = publish_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=run.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
    )
    mark_run_complete(run, parent_group=None, run_name="r1")
    mutations = (
        (REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR, 9),
        (REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR, "alien_policy"),
        (
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
            {"policy": "alien", "publication_owner": "f" * 32},
        ),
        ("latest_pending", "alien_pending"),
    )
    for name, value in mutations:
        parent.attrs[name] = copy.deepcopy(value)
        with pytest.raises(
            RefinedSubjectMaskCoordinatePublicationError,
            match="concurrent parent mutation|rollback was incomplete",
        ):
            _activate_validated_refined_subject_mask_coordinate_surfaces(
                root,
                parent,
                pending,
                run_name="r1",
                publication_owner_token=run.attrs[
                    REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
                ],
                selector_snapshot=snapshot,
            )
        if name in parent.attrs:
            del parent.attrs[name]
        parent.attrs["latest_pending"] = "r1"
    assert run.attrs["stage_selector_eligible"] is False


def test_refined_activation_detects_and_preserves_all_alien_interwrite_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run, _raw, snapshot = _refined_fixture(monkeypatch)
    _prepare(root, run)
    pending = publish_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=run.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
    )
    mark_run_complete(run, parent_group=None, run_name="r1")
    alien_lease: object = {
        "schema_id": "palette.refined_subject_mask_publication_lease",
        "schema_version": 1,
        "policy": module._PUBLICATION_POLICY,
        "run_path": "refined_subject_masks_runs/alien",
        "publication_owner": "a" * 32,
        "base_generation": 0,
        "next_generation": 1,
    }

    attacks = (
        (
            REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
            91,
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        ),
        (
            REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
            "alien_policy",
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        ),
        (
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
            alien_lease,
            "latest_complete",
        ),
        (
            "latest_pending",
            "alien_pending",
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        ),
    )

    for attacked_name, alien_value, trigger_name in attacks:
        class MutateBetweenWrites(dict):
            attacked = False

            def __setitem__(self, key: str, value: Any) -> None:
                super().__setitem__(key, value)
                if key == trigger_name and not self.attacked:
                    self.attacked = True
                    super().__setitem__(attacked_name, copy.deepcopy(alien_value))

        parent.attrs = MutateBetweenWrites(parent.attrs)
        with pytest.raises(RefinedSubjectMaskCoordinatePublicationError):
            _activate_validated_refined_subject_mask_coordinate_surfaces(
                root,
                parent,
                pending,
                run_name="r1",
                publication_owner_token=run.attrs[
                    REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
                ],
                selector_snapshot=snapshot,
            )
        assert parent.attrs[attacked_name] == alien_value
        assert "latest_complete" not in parent.attrs
        assert "latest" not in parent.attrs
        assert run.attrs["stage_selector_eligible"] is False

        restored = dict(parent.attrs)
        for name, (present, value) in snapshot.items():
            if present:
                restored[name] = copy.deepcopy(value)
            else:
                restored.pop(name, None)
        restored["latest_pending"] = "r1"
        parent.attrs = restored
    assert run.attrs["stage_selector_eligible"] is False


def test_refined_publication_binds_component_qc_presence_and_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, run, _raw, snapshot = _refined_fixture(monkeypatch)
    qc = run["components"]["subject_body"].create_group("qc")
    qc.attrs["schema_id"] = "refined_subject_body_mask_qc"
    qc.create_array(
        "requires_review",
        data=np.asarray([False, True], dtype=bool),
        chunks=(2,),
    )

    loaded = _publish_activate(root, parent, run, snapshot)

    components = loaded.component_qc_inventory.record["components"]
    assert components["subject_body"]["status"] == "present"
    assert components["eyes_union"]["status"] == "absent"
    assert components["swim_bladder"]["status"] == "absent"
    assert (
        loaded.inventory.record["component_qc_inventory"]["record_sha256"]
        == loaded.component_qc_inventory.record_sha256
    )
    qc["requires_review"].data[0] = True
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="component QC inventory",
    ):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
        )


def test_refined_optional_geometry_inventory_is_closed_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    run["components"]["subject_body"]["geometry"].create_array(
        "undocumented_xy",
        data=np.zeros((2, 2), dtype=np.float32),
        chunks=(2, 2),
    )
    _prepare(root, run)

    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="closed-world geometry container",
    ):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )


def test_refined_component_and_relation_inventory_is_closed_world_but_allows_classified_non_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    _prepare(root, run)
    components = run["components"]
    components.create_group("undeclared_component")
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="undeclared component namespaces",
    ):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )
    components.children.pop("undeclared_component")

    component = components["subject_body"]
    component.create_array(
        "mystery_xy",
        data=np.zeros((2, 2), dtype=np.float32),
        chunks=(2, 2),
    )
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="undocumented root array",
    ):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )
    component.children.pop("mystery_xy")

    relations = (
        run["relations"] if "relations" in run else run.create_group("relations")
    )
    relations.create_group("unknown_relation")
    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="unknown relation namespaces",
    ):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )
    relations.children.pop("unknown_relation")

    classified = component.create_array(
        "categorical_quality_code",
        data=np.asarray([1, 2], dtype=np.int16),
        chunks=(2,),
    )
    classified.attrs["surface_role"] = "explicit_non_geometry"
    classified.attrs["geometry_semantics"] = "none"
    published = publish_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=run.attrs[
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
        ],
    )
    classified_inventory = published.inventory.record["closed_world_structure"][
        "components"
    ]["subject_body"]["arrays"]["categorical_quality_code"]
    assert classified_inventory["classification"] == "explicit_non_geometry"


def test_refined_optional_invalid_geometry_requires_nan_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _parent, run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    run["components"]["subject_body"]["sampled_contours"]["points_xy"].data[
        1
    ] = 0.0
    _prepare(root, run)

    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="all-NaN sentinel",
    ):
        publish_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/r1",
            expected_publication_owner=run.attrs[
                REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR
            ],
        )


def test_strict_loader_rejects_unsupported_legacy_refined_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, parent, _run, _raw, _snapshot_value = _refined_fixture(monkeypatch)
    legacy = parent.create_group("legacy")
    legacy.attrs.update(
        {
            "mask_labels": ["subject_body"],
            "stage_selector_eligible": True,
        }
    )
    mark_run_started(legacy, run_name="legacy", stage="refine_subject_masks")
    mark_run_complete(legacy, parent_group=None, run_name="legacy")
    legacy.create_array(
        "masks_roi",
        data=np.zeros((1, 1, 4, 4), dtype=np.uint8),
        chunks=(1, 1, 4, 4),
    )

    with pytest.raises(
        RefinedSubjectMaskCoordinatePublicationError,
        match="publication owner",
    ):
        load_persisted_refined_subject_mask_coordinate_surfaces(
            root,
            "refined_subject_masks_runs/legacy",
        )
