from __future__ import annotations

import copy
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pytest
import zarr

from fisheye.analysis import subject_shape_runs as subject_shape_writer
from fisheye.analysis_workflows.materializers import subject_shape as shape_materializer
import fisheye.shared.subject_shape_coordinate_publication as module
from fisheye.shared.coordinate_descriptor import COORDINATE_DESCRIPTOR_ATTR
from fisheye.shared.mask_geometry import batch_mask_spatial_metrics
from fisheye.shared.model_input_transform import resolve_model_input_transform
from fisheye.shared.refined_subject_mask_coordinate_publication import (
    REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
    REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
    REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
    _activate_validated_refined_subject_mask_coordinate_surfaces,
    prepare_refined_subject_mask_coordinate_context,
    publish_refined_subject_mask_coordinate_surfaces,
)
from fisheye.shared.refined_subject_mask_mutation import (
    stamp_refined_subject_mask_editable_draft,
)
from fisheye.shared.subject_mask_coordinate_publication import (
    SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
    SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    SUBJECT_MASK_PUBLICATION_OWNER_ATTR,
    SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
    _activate_validated_subject_mask_coordinate_surfaces,
    _load_completed_ineligible_subject_mask_coordinate_surfaces,
    load_persisted_subject_mask_crop_source,
    prepare_subject_mask_coordinate_context,
    publish_subject_mask_coordinate_surfaces,
    selected_subject_mask_crop_values,
)
from fisheye.shared.subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR,
    SubjectShapeCoordinatePublicationError,
    activate_subject_shape_coordinate_publication,
    deep_audit_subject_shape_payload_receipt,
    load_persisted_subject_shape_coordinate_publication,
    require_translation_only_refined_placement,
    selector_snapshot,
)
from fisheye.shared.zarr_run_completion import mark_run_complete, mark_run_started
from fisheye.shared.zarr.subject_mask_schema import (
    derive_subject_mask_frame_row_offsets,
)
from tests.unit.fisheye.test_keypoint_coordinate_publication import (
    _real_canonical_archive,
)
from tests.unit.fisheye.subject_shape_test_fixtures import (
    resolve_canonical_refined_archive_template,
)


LABELS = ("subject_body", "swim_bladder", "eye_left", "eye_right")
MODEL_ARTIFACT = {
    "role": "subject_mask_unet_checkpoint",
    "path": "/models/subject-mask.pt",
    "fingerprint_scheme": "content_v1",
    "sha256": "a" * 64,
    "size_bytes": 123,
    "mtime_ns": 456,
    "source": "computed",
}


def _patch_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject_shape_writer,
        "get_git_info",
        lambda repo_path=None: {  # noqa: ARG005
            "commit_hash": "c" * 40,
            "short_hash": "cccccccc",
            "branch": "main",
            "is_dirty": False,
            "remote_url": "git@example.com:palette.git",
        },
    )
    monkeypatch.setattr(
        subject_shape_writer,
        "get_environment_info",
        lambda **kwargs: {  # noqa: ARG005
            "environment": {"python": "3.11"},
            "platform": {
                "hostname": "shape-host",
                "system": "Linux",
                "release": "6.8",
                "python_version": "3.11.0",
                "machine": "x86_64",
            },
        },
    )


def _snapshot(parent: Any, *, refined: bool) -> dict[str, tuple[bool, Any]]:
    if refined:
        lifecycle = (
            REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
            REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        )
    else:
        lifecycle = (
            SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
            SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
            SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        )
    return {
        name: (name in parent.attrs, copy.deepcopy(parent.attrs.get(name)))
        for name in (
            "latest",
            "latest_complete",
            "latest_pending",
            "authoritative_run",
            "authoritative_run_provenance",
            *lifecycle,
        )
    }


def _fish_masks() -> np.ndarray:
    masks = np.zeros((2, len(LABELS), 40, 40), dtype=np.uint8)
    for row, shift in enumerate((0, 2)):
        masks[row, 0, 10 + shift : 33 + shift, 15:25] = 1
        masks[row, 1, 21 + shift : 25 + shift, 18:22] = 1
        masks[row, 2, 6 + shift : 11 + shift, 11:16] = 1
        masks[row, 3, 6 + shift : 11 + shift, 24:29] = 1
    # Exercise the canonical invalid-row rule: the second row lacks one
    # anatomical side anchor and must publish all-NaN body-frame geometry.
    masks[1, 3] = 0
    return masks


def _create_canonical_subject_masks(root: Any) -> Any:
    parent = root.require_group("subject_mask_runs")
    run = parent.create_group("s1")
    owner = uuid4().hex
    transform = resolve_model_input_transform((40, 40), mode="identity")
    run.attrs.update(
        {
            "stage_selector_eligible": False,
            SUBJECT_MASK_PUBLICATION_OWNER_ATTR: owner,
            "mask_labels": list(LABELS),
            "model_input_transform": transform.to_attrs(),
            "mask_probability_threshold": 0.5,
            "output_semantics": "multilabel",
            "overlap_policy": "independent_sigmoid",
            "probability_semantics": "sigmoid_multilabel_logits",
            "probabilities_dtype": "uint8",
            "probabilities_encoding": "linear_uint8_0_255",
            "masks_roi_materialized": True,
            "binary_masks_materialized": True,
            "binary_masks_source": "threshold(mask_probs_roi, threshold=0.5)",
            "bbox_xyxy_convention": "pixel_edge_half_open",
            "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
            "source_checkpoint": MODEL_ARTIFACT["path"],
            "subject_mask_model_artifact": copy.deepcopy(MODEL_ARTIFACT),
            "provenance": {
                "stage": "subject_masks",
                "method": "unit_test_canonical_raw",
                "source": "crop_runs/c1",
            },
        }
    )
    mark_run_started(run, run_name="s1", stage="subject_masks")
    source = load_persisted_subject_mask_crop_source(root, "crop_runs/c1")
    selected = selected_subject_mask_crop_values(
        source,
        np.asarray([1, 0], dtype="<i8"),
    )
    for name in (
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "source_crop_xywh",
    ):
        run.create_array(name, data=selected[name])
    source_frames = np.asarray(
        selected["source_acquisition_frame_index"], dtype=np.int64
    )
    run.create_array(
        "frame_row_offsets",
        data=derive_subject_mask_frame_row_offsets(
            source_frames,
            n_frames=int(source_frames.max(initial=-1)) + 1,
        ),
    )

    masks = _fish_masks()
    probabilities = masks * np.uint8(255)
    metrics_values = batch_mask_spatial_metrics(masks)
    run.create_array("mask_probs_roi", data=probabilities)
    run.create_array("masks_roi", data=masks)
    run.create_array(
        "available_channels",
        data=np.ones((len(LABELS),), dtype=bool),
    )
    metrics = run.create_group("metrics")
    metrics.create_array(
        "prob_max",
        data=probabilities.max(axis=(-2, -1)).astype(np.float32) / 255.0,
    )
    for name, values in metrics_values.items():
        metrics.create_array(name, data=values)

    prepare_subject_mask_coordinate_context(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=owner,
        crop_path="crop_runs/c1",
        mask_labels=LABELS,
        model_input_transform=transform,
        model_artifact=MODEL_ARTIFACT,
        mask_probability_threshold=0.5,
    )
    publish_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=owner,
    )
    # Zarr v3 attribute mappings are snapshot-like.  Re-resolve the child after
    # publication before adding completion attrs so an older handle cannot
    # replace the freshly stamped coordinate contract.
    run = root["subject_mask_runs/s1"]
    parent = root["subject_mask_runs"]
    snapshot = _snapshot(parent, refined=False)
    parent.attrs["latest_pending"] = "s1"
    mark_run_complete(run, parent_group=None, run_name="s1")
    proof = _load_completed_ineligible_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=owner,
    )
    _activate_validated_subject_mask_coordinate_surfaces(
        root,
        parent,
        proof,
        run_name="s1",
        publication_owner_token=owner,
        selector_snapshot=snapshot,
    )
    return run


def _create_canonical_refined_masks(root: Any, raw: Any) -> Any:
    parent = root.require_group("refined_subject_masks_runs")
    run = parent.create_group("r1")
    owner = uuid4().hex
    run.attrs.update(
        {
            "stage_selector_eligible": False,
            REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR: owner,
            "source_subject_mask_run": "s1",
            "mask_labels": list(LABELS),
            "label_schema_id": "subject_v1_left_right",
            "component_metrics_schema_id": "refined_subject_component_mask_metrics_v1",
            "component_review_statuses": {
                name: {"state": "approved", "method": "unit_test"}
                for name in LABELS
            },
            "refined_subject_mask_review_status": {
                "state": "approved",
                "method": "unit_test",
            },
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
    stamp_refined_subject_mask_editable_draft(run)
    mark_run_started(run, run_name="r1", stage="refine_subject_masks")
    for name in (
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "frame_row_offsets",
        "source_crop_xywh",
        "available_channels",
        "masks_roi",
    ):
        run.create_array(name, data=np.asarray(raw[name][:]))
    metrics = run.create_group("metrics")
    for name in (
        "mask_present",
        "area_px",
        "centroid_xy",
        "centroid_valid",
        "bbox_xyxy",
        "bbox_valid",
    ):
        metrics.create_array(name, data=np.asarray(raw[f"metrics/{name}"][:]))
    run.attrs.update(
        {
            "derived_mask_caches_stale": False,
            "metrics_stale": False,
            "contours_stale": False,
        }
    )
    components = run.create_group("components")
    for component_index, component_name in enumerate(LABELS):
        component = components.create_group(component_name)
        component.create_array(
            "area_px",
            data=np.asarray(metrics["area_px"][:, component_index]),
        )
        component.create_array(
            "mask_present",
            data=np.asarray(metrics["mask_present"][:, component_index]),
        )
        provenance = component.create_group("provenance")
        source_path = "subject_mask_runs/s1/mask_probs_roi"
        provenance.attrs.update(
            {
                "source_channels": [component_name],
                "source_surface_path": source_path,
                "source_surface_kind": "probability",
                "source_probability_path": source_path,
                "source_probability_encoding": raw.attrs["probabilities_encoding"],
                "source_probability_threshold": float(
                    raw.attrs["mask_probability_threshold"]
                ),
                "source_binary_derivation": "smart_finalize(mask_probs_roi)",
                "finalization_method": "smart_finalize_subject_masks_v1",
                "finalization_policy": {"fixture": "exact_source_selection_v1"},
            }
        )

    prepare_refined_subject_mask_coordinate_context(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=owner,
        source_subject_mask_path="subject_mask_runs/s1",
        mask_labels=LABELS,
    )
    proof = publish_refined_subject_mask_coordinate_surfaces(
        root,
        "refined_subject_masks_runs/r1",
        expected_publication_owner=owner,
    )
    run = root["refined_subject_masks_runs/r1"]
    parent = root["refined_subject_masks_runs"]
    snapshot = _snapshot(parent, refined=True)
    parent.attrs["latest_pending"] = "r1"
    mark_run_complete(run, parent_group=None, run_name="r1")
    _activate_validated_refined_subject_mask_coordinate_surfaces(
        root,
        parent,
        proof,
        run_name="r1",
        publication_owner_token=owner,
        selector_snapshot=snapshot,
    )
    return run


def _canonical_refined_archive(tmp_path: Any) -> tuple[Any, Any]:
    root, _keypoint_run = _real_canonical_archive(tmp_path)
    raw = _create_canonical_subject_masks(root)
    return root, _create_canonical_refined_masks(root, raw)


@pytest.fixture(scope="session")
def canonical_refined_template() -> Path:
    return resolve_canonical_refined_archive_template()


@pytest.fixture
def canonical_refined_archive(
    tmp_path: Path,
    canonical_refined_template: Path,
) -> tuple[zarr.Group, zarr.Group]:
    destination = tmp_path / "canonical.zarr"
    shutil.copytree(canonical_refined_template, destination)
    root = zarr.open_group(str(destination), mode="r+", use_consolidated=False)
    return root, root["refined_subject_masks_runs/r1"]


@pytest.fixture(scope="session")
def canonical_subject_shape_profile_template(
    tmp_path_factory: pytest.TempPathFactory,
    canonical_refined_template: Path,
) -> Path:
    destination = tmp_path_factory.mktemp("subject-shape-profile") / "canonical.zarr"
    shutil.copytree(canonical_refined_template, destination)
    root = zarr.open_group(str(destination), mode="r+", use_consolidated=False)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _patch_provenance(monkeypatch)
        subject_shape_writer.write_subject_shape_run_group(
            root,
            refined_run="r1",
            run_name="shape_profile_attack",
            chunk_size=2,
        )
    return destination


@pytest.fixture
def canonical_subject_shape_profile_root(
    tmp_path: Path,
    canonical_subject_shape_profile_template: Path,
) -> zarr.Group:
    destination = tmp_path / "canonical-subject-shape.zarr"
    shutil.copytree(canonical_subject_shape_profile_template, destination)
    return zarr.open_group(str(destination), mode="r+", use_consolidated=False)


def _restamp_tampered_subject_shape_manifest(
    run: Any,
    *,
    profile_field: str,
    value: Any,
    array_paths: tuple[str, ...] = (),
) -> str:
    """Recompute the outer digest so exact-profile tests are not stale-hash tests."""

    manifest = copy.deepcopy(run.attrs[module.SUBJECT_SHAPE_MANIFEST_ATTR])
    manifest["schema_inventory"]["maintained_profile"][profile_field] = value
    for path in array_paths:
        manifest["arrays"][path] = module._array_record(path, run[path])
        manifest["schema_inventory"]["arrays"][path] = {
            "role": "compatibility_row_lineage"
        }
    digest = module._canonical_sha256(manifest)
    run.attrs[module.SUBJECT_SHAPE_MANIFEST_ATTR] = manifest
    run.attrs[f"{module.SUBJECT_SHAPE_MANIFEST_ATTR}_sha256"] = digest
    run.attrs["publication_manifest_sha256"] = digest
    return digest


@pytest.mark.parametrize(
    ("variant", "expected_error"),
    (
        ("component_reorder", "component order"),
        ("relation_reduction", "relation order"),
        ("optional_lineage", "row_index bundle"),
        ("row_index_dtype", "rank-1 uint64"),
        (
            "row_index_value",
            "payload receipt validation|canonical direct row-identity array",
        ),
        ("schema_version", "run identity"),
    ),
)
def test_maintained_subject_shape_profile_rejects_self_declared_variants_with_recomputed_digest(
    canonical_subject_shape_profile_root: zarr.Group,
    variant: str,
    expected_error: str,
) -> None:
    root = canonical_subject_shape_profile_root
    run = root["analysis/subject_shape_runs/shape_profile_attack"]

    if variant == "component_reorder":
        reordered = ["swim_bladder", "subject_body", "eye_left", "eye_right"]
        run.attrs["component_names"] = reordered
        digest = _restamp_tampered_subject_shape_manifest(
            run,
            profile_field="component_order",
            value=reordered,
        )
    elif variant == "relation_reduction":
        reduced = ["eye_pair", "swim_bladder_to_body"]
        run.attrs["relation_names"] = reduced
        digest = _restamp_tampered_subject_shape_manifest(
            run,
            profile_field="relation_order",
            value=reduced,
        )
    elif variant == "optional_lineage":
        frame_indices = run["source_acquisition_frame_index"]
        run["row_index"].create_array(
            "frame_indices",
            data=np.asarray(frame_indices[:], dtype=np.int32),
        )
        copied = ["frame_indices", "source_crop_row_ids", "instance_key"]
        run.attrs["row_lineage_copied"] = copied
        run.attrs["row_lineage_missing"] = [
            "detection_indices",
            "source_refined_row_ids",
            "source_detect_row_index",
        ]
        digest = _restamp_tampered_subject_shape_manifest(
            run,
            profile_field="row_index_arrays",
            value=copied,
            array_paths=("row_index/frame_indices",),
        )
    elif variant == "row_index_dtype":
        row_index = run["row_index"]
        values = np.asarray(row_index["instance_key"][:], dtype=np.int64)
        del row_index["instance_key"]
        row_index.create_array("instance_key", data=values)
        digest = _restamp_tampered_subject_shape_manifest(
            run,
            profile_field="row_index_arrays",
            value=list(module.CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS),
            array_paths=("row_index/instance_key",),
        )
    elif variant == "row_index_value":
        row_index = run["row_index"]
        values = np.asarray(row_index["instance_key"][:], dtype=np.uint64)
        values[0] = values[0] + np.uint64(1)
        row_index["instance_key"][:] = values
        digest = _restamp_tampered_subject_shape_manifest(
            run,
            profile_field="row_index_arrays",
            value=list(module.CANONICAL_SUBJECT_SHAPE_ROW_INDEX_ARRAYS),
            array_paths=("row_index/instance_key",),
        )
    else:
        run.attrs["schema_version"] = 999
        digest = _restamp_tampered_subject_shape_manifest(
            run,
            profile_field="run_schema_version",
            value=999,
        )

    assert run.attrs["publication_manifest_sha256"] == digest
    with pytest.raises(SubjectShapeCoordinatePublicationError, match=expected_error):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_profile_attack",
        )


def test_subject_shape_writer_publishes_exact_source_camera_geometry_and_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_refined_archive: tuple[zarr.Group, zarr.Group],
) -> None:
    _patch_provenance(monkeypatch)
    root, refined = canonical_refined_archive
    roi_metrics = batch_mask_spatial_metrics(np.asarray(refined["masks_roi"][:]))
    offsets = np.asarray(refined["source_crop_xywh"][:, :2], dtype=np.float32)

    summary = subject_shape_writer.write_subject_shape_run_group(
        root,
        refined_run="r1",
        run_name="shape_001",
        chunk_size=2,
    )

    assert summary["status"] == "updated"
    parent = root["analysis/subject_shape_runs"]
    run = parent["shape_001"]
    assert parent.attrs["latest"] == "shape_001"
    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["schema_version"] == 4
    assert run.attrs["method_version"] == 11
    assert run.attrs["coordinate_binding_status"] == "bound_canonical_v2"
    loaded = load_persisted_subject_shape_coordinate_publication(
        root,
        "analysis/subject_shape_runs/shape_001",
    )
    payload_run_path = (
        tmp_path / "canonical.zarr/analysis/subject_shape_runs/shape_001"
    )
    deep_audit = deep_audit_subject_shape_payload_receipt(
        run,
        payload_run_path=payload_run_path,
        hash_workers=1,
    )
    assert deep_audit["valid"] is True
    assert deep_audit["physical_rehash_performed"] is True

    def reject_live_array_hash(_node: Any) -> str:
        raise AssertionError("receipt-backed load performed a live array hash")

    with monkeypatch.context() as receipt_loader_patch:
        receipt_loader_patch.setattr(
            module,
            "array_payload_sha256",
            reject_live_array_hash,
        )
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )

    array_type = type(run["instance_key"])

    def reject_array_decode(_self: Any, _selection: Any) -> Any:
        raise AssertionError("metadata-only validation decoded an array")

    with monkeypatch.context() as metadata_only_patch:
        metadata_only_patch.setattr(array_type, "__getitem__", reject_array_decode)
        metadata_proof = module.validate_sealed_subject_shape_publication_metadata(
            root,
            "analysis/subject_shape_runs/shape_001",
            expected_selector_eligible=True,
            expected_publication_owner=run.attrs[
                SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR
            ],
            payload_run_path=payload_run_path,
        )
    assert metadata_proof.row_count == 2
    assert metadata_proof.manifest.record_sha256 == run.attrs[
        "publication_manifest_sha256"
    ]

    body = run["components/subject_body"]
    original_point_space = body.attrs["point_coordinate_space"]
    body.attrs["point_coordinate_space"] = "tampered"
    with pytest.raises(
        SubjectShapeCoordinatePublicationError,
        match="closed manifest",
    ):
        module.validate_sealed_subject_shape_publication_metadata(
            root,
            "analysis/subject_shape_runs/shape_001",
            expected_selector_eligible=True,
            expected_publication_owner=run.attrs[
                SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR
            ],
            payload_run_path=payload_run_path,
        )
    body.attrs["point_coordinate_space"] = original_point_space

    validation_attr = module.SUBJECT_SHAPE_PAYLOAD_VALIDATION_RECEIPT_ATTR
    original_validation = copy.deepcopy(run.attrs[validation_attr])
    tampered_validation = copy.deepcopy(original_validation)
    tampered_validation["numerical_policy"]["normal_load_physical_rehash"] = True
    tampered_validation["numerical_policy_sha256"] = module._canonical_sha256(
        tampered_validation["numerical_policy"]
    )
    tampered_validation["record_sha256"] = module._canonical_sha256(
        {
            key: value
            for key, value in tampered_validation.items()
            if key != "record_sha256"
        }
    )
    run.attrs[validation_attr] = tampered_validation
    with pytest.raises(SubjectShapeCoordinatePublicationError, match="policy"):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    run.attrs[validation_attr] = original_validation

    integrity_attr = module.SUBJECT_SHAPE_PAYLOAD_INTEGRITY_RECEIPT_ATTR
    profile_attr = module.SUBJECT_SHAPE_PAYLOAD_RECEIPT_PROFILE_ATTR
    original_integrity = copy.deepcopy(run.attrs[integrity_attr])
    original_profile = run.attrs[profile_attr]
    del run.attrs[integrity_attr]
    del run.attrs[validation_attr]
    with monkeypatch.context() as receipt_guard:
        receipt_guard.setattr(
            module,
            "_iter_arrays",
            lambda *_args, **_kwargs: pytest.fail(
                "receipt-free load reached scientific array traversal"
            ),
        )
        with pytest.raises(
            SubjectShapeCoordinatePublicationError,
            match="complete sealed payload receipt pair",
        ):
            load_persisted_subject_shape_coordinate_publication(
                root,
                "analysis/subject_shape_runs/shape_001",
            )
        del run.attrs[profile_attr]
        with pytest.raises(
            SubjectShapeCoordinatePublicationError,
            match="receipt-free publications are unsupported",
        ):
            load_persisted_subject_shape_coordinate_publication(
                root,
                "analysis/subject_shape_runs/shape_001",
            )
    run.attrs[integrity_attr] = original_integrity
    run.attrs[validation_attr] = original_validation
    run.attrs[profile_attr] = original_profile
    np.testing.assert_array_equal(run["instance_key"][:], refined["instance_key"][:])
    np.testing.assert_array_equal(
        run["source_crop_row_ids"][:],
        refined["source_crop_row_ids"][:],
    )

    body_index = LABELS.index("subject_body")
    expected_centroid = roi_metrics["centroid_xy"][:, body_index] + offsets
    expected_bbox = roi_metrics["bbox_xyxy"][:, body_index] + np.tile(offsets, (1, 2))
    body = run["components/subject_body"]
    np.testing.assert_allclose(body["centroid_xy"][:], expected_centroid)
    np.testing.assert_allclose(body["bbox_xyxy"][:], expected_bbox)
    centroid_descriptor = loaded.descriptors[
        "components/subject_body/centroid_xy"
    ].descriptor
    bbox_descriptor = loaded.descriptors[
        "components/subject_body/bbox_xyxy"
    ].descriptor
    centerline_descriptor = loaded.descriptors[
        "components/subject_body/centerline_xy"
    ].descriptor
    aggregate_descriptor = loaded.descriptors["component_centroid_xy"].descriptor
    assert aggregate_descriptor.geometry_type == "point_xy"
    assert aggregate_descriptor.collection_axis is not None
    assert centroid_descriptor.geometry_type == "point_xy"
    assert centroid_descriptor.collection_axis is None
    assert centroid_descriptor.source_camera_overlay.status == "direct"
    assert centroid_descriptor.reference_extent.width == 100
    assert centroid_descriptor.reference_extent.height == 80
    assert bbox_descriptor.geometry_type == "bbox_xyxy"
    assert bbox_descriptor.pixel_convention == "pixel_edge_half_open"
    assert centerline_descriptor.geometry_type == "polyline_xy"
    assert run.attrs["roi_local_point_arrays_retained"] is False

    frame = run["body_frame"]
    assert frame["axis_valid"][:].tolist() == [True, False]
    assert bool(np.all(np.isfinite(frame["origin_xy"][0])))
    assert bool(np.all(np.isfinite(frame["forward_axis_xy"][0])))
    assert float(frame["forward_axis_xy"][0, 1]) < 0.0
    assert bool(np.all(np.isnan(frame["origin_xy"][1])))
    assert bool(np.all(np.isnan(frame["forward_axis_xy"][1])))
    assert bool(np.all(np.isnan(frame["left_axis_xy"][1])))
    assert loaded.descriptors["body_frame/origin_xy"].descriptor.geometry_type == "point_xy"
    assert loaded.descriptors["body_frame/forward_axis_xy"].descriptor.geometry_type == "vector_xy"
    assert loaded.descriptors[
        "body_frame/forward_axis_xy"
    ].descriptor.source_camera_overlay.status == "not_suitable"
    principal_axis = loaded.descriptors[
        "components/subject_body/principal_axis_xy"
    ].descriptor
    assert principal_axis.geometry_type == "vector_xy"
    assert principal_axis.profile_id == "source_camera_image_px.unit_vector_y_down.v1"
    tail_tangent = loaded.descriptors[
        "components/subject_body/tail_tangent_xy"
    ].descriptor
    assert tail_tangent.geometry_type == "vector_sequence_xy"
    assert tail_tangent.component_units == ("unitless", "unitless")
    assert tail_tangent.pixel_convention == "not_applicable"
    assert tail_tangent.source_camera_overlay.status == "not_suitable"
    assert loaded.tail_sample_axis.record["sample_direction"] == (
        "tail_base_to_tail_tip"
    )
    assert loaded.tail_sample_axis.record["cardinality"] == (
        subject_shape_writer.TAIL_SAMPLE_COUNT
    )
    curvature_binding = loaded.require_scalar_surface(
        "components/subject_body/tail_curvature_px_inv",
        units="px^-1",
        surface_kind="row_profile",
    )
    curvature_record = curvature_binding.semantics.record
    assert curvature_record["profile_axis"]["axis_record"] == {
        "record_ref": loaded.tail_sample_axis.record_ref,
        "record_sha256": loaded.tail_sample_axis.record_sha256,
    }
    assert curvature_record["profile_axis"]["cardinality"] == (
        subject_shape_writer.TAIL_SAMPLE_COUNT
    )
    assert curvature_record["validity"]["surface"]["relative_ref"] == (
        "components/subject_body/tail_sample_valid"
    )
    assert "positive_is_clockwise" in curvature_record["sign_convention"]
    assert loaded.derivation.record["scalar_surface_inventory"] == {
        "record_ref": loaded.scalar_surface_inventory.record_ref,
        "record_sha256": loaded.scalar_surface_inventory.record_sha256,
    }
    assert loaded.manifest.record["scalar_surfaces"][
        "components/subject_body/tail_curvature_px_inv"
    ] == {
        "record_ref": curvature_binding.semantics.record_ref,
        "record_sha256": curvature_binding.semantics.record_sha256,
    }
    eye_offset = loaded.descriptors[
        "relations/eyes_to_body/left_eye_offset_xy"
    ].descriptor
    assert eye_offset.profile_id == (
        "source_camera_image_px.displacement_vector_y_down.v1"
    )
    assert eye_offset.source_camera_overlay.status == "not_suitable"
    assert loaded.temporal_authority.record_ref.endswith(
        "@source_row_temporal_authority"
    )
    assert loaded.derivation.record["source_refined_subject_masks"][
        "component_qc_inventory"
    ]["record_sha256"] == loaded.source.component_qc_inventory.record_sha256
    assert loaded.scientific_configuration.record["run_attrs"]["method_version"] == 11
    assert loaded.scientific_configuration.record["maintained_profile"] == (
        module.subject_shape_maintained_profile_record()
    )
    assert loaded.manifest.record["schema_inventory"]["maintained_profile"] == (
        module.subject_shape_maintained_profile_record()
    )
    assert loaded.heading_semantics.record["formula"] == (
        "degrees(atan2(-forward_y, forward_x))"
    )

    # Immutable output payloads are guarded by the explicit deep-audit path;
    # normal loads use the sealed receipt. Array descriptors and exact refined
    # authority remain ordinary live metadata gates; out-of-band payload
    # mutation, including identity arrays, is an explicit custody audit.
    original = float(body["centroid_xy"][0, 0])
    body["centroid_xy"][0, 0] = original + 1.0
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        deep_audit_subject_shape_payload_receipt(
            run,
            payload_run_path=payload_run_path,
            hash_workers=1,
        )
    body["centroid_xy"][0, 0] = original

    descriptor = copy.deepcopy(body["centroid_xy"].attrs[COORDINATE_DESCRIPTOR_ATTR])
    del body["centroid_xy"].attrs[COORDINATE_DESCRIPTOR_ATTR]
    with pytest.raises(ValueError):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    body["centroid_xy"].attrs[COORDINATE_DESCRIPTOR_ATTR] = descriptor

    original_key = np.asarray(run["instance_key"][:]).copy()
    run["instance_key"][:] = original_key[::-1]
    load_persisted_subject_shape_coordinate_publication(
        root,
        "analysis/subject_shape_runs/shape_001",
    )
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        deep_audit_subject_shape_payload_receipt(
            run,
            payload_run_path=payload_run_path,
            hash_workers=1,
        )
    run["instance_key"][:] = original_key

    run.attrs["coordinate_binding_status"] = "publishing_canonical_binding_v1"
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    run.attrs["coordinate_binding_status"] = "bound_canonical_v2"

    tail_s = np.asarray(body["tail_sample_s"][:]).copy()
    body["tail_sample_s"][1] = np.float32(0.123)
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        deep_audit_subject_shape_payload_receipt(
            run,
            payload_run_path=payload_run_path,
            hash_workers=1,
        )
    body["tail_sample_s"][:] = tail_s

    curvature_node = body["tail_curvature_px_inv"]
    scalar_attr = module.SUBJECT_SHAPE_SCALAR_SURFACE_ATTR
    scalar_digest_attr = f"{scalar_attr}_sha256"
    original_scalar_record = copy.deepcopy(curvature_node.attrs[scalar_attr])
    original_scalar_digest = curvature_node.attrs[scalar_digest_attr]
    semantic_tampers = (
        (
            "array ref",
            lambda record: record["surface"].__setitem__(
                "array_ref",
                "/analysis/subject_shape_runs/alien/tail_curvature_px_inv",
            ),
        ),
        (
            "cardinality",
            lambda record: record["profile_axis"].__setitem__(
                "cardinality",
                int(record["profile_axis"]["cardinality"]) + 1,
            ),
        ),
        (
            "unsupported schema",
            lambda record: record.__setitem__("schema_version", 999),
        ),
    )
    for _label, mutate in semantic_tampers:
        tampered = copy.deepcopy(original_scalar_record)
        mutate(tampered)
        curvature_node.attrs[scalar_attr] = tampered
        curvature_node.attrs[scalar_digest_attr] = module._canonical_sha256(tampered)
        with pytest.raises(SubjectShapeCoordinatePublicationError):
            load_persisted_subject_shape_coordinate_publication(
                root,
                "analysis/subject_shape_runs/shape_001",
            )
        curvature_node.attrs[scalar_attr] = copy.deepcopy(original_scalar_record)
        curvature_node.attrs[scalar_digest_attr] = original_scalar_digest

    curvature_values = np.asarray(curvature_node[:]).copy()
    finite_curvature = np.argwhere(np.isfinite(curvature_values))
    assert finite_curvature.size > 0
    curvature_row, curvature_sample = (
        int(finite_curvature[0, 0]),
        int(finite_curvature[0, 1]),
    )
    curvature_node[curvature_row, curvature_sample] = np.float32(
        curvature_values[curvature_row, curvature_sample] + 0.25
    )
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        deep_audit_subject_shape_payload_receipt(
            run,
            payload_run_path=payload_run_path,
            hash_workers=1,
        )
    curvature_node[:] = curvature_values

    tail_valid = np.asarray(body["tail_sample_valid"][:], dtype=bool).copy()
    body["tail_sample_valid"][0] = ~tail_valid[0]
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        deep_audit_subject_shape_payload_receipt(
            run,
            payload_run_path=payload_run_path,
            hash_workers=1,
        )
    body["tail_sample_valid"][:] = tail_valid

    original_method = body.attrs["centerline_method"]
    body.attrs["centerline_method"] = "tampered"
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    body.attrs["centerline_method"] = original_method

    original_mask = np.asarray(refined["masks_roi"][0, 0]).copy()
    refined["masks_roi"][0, 0, 0, 0] = np.uint8(1)
    with pytest.raises(SubjectShapeCoordinatePublicationError):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    refined["masks_roi"][0, 0] = original_mask

    body.create_array(
        "rogue_point_xy",
        data=np.zeros((2, 2), dtype=np.float32),
    )
    with pytest.raises(SubjectShapeCoordinatePublicationError, match="array inventory"):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    del body["rogue_point_xy"]

    body.create_group("rogue_empty_group")
    with pytest.raises(SubjectShapeCoordinatePublicationError, match="group inventory"):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    del body["rogue_empty_group"]

    run.attrs["coordinate_space"] = "roi_local_px"
    with pytest.raises(
        SubjectShapeCoordinatePublicationError,
        match="outside the controlled vocabulary",
    ):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    del run.attrs["coordinate_space"]

    body.attrs["coordinate_space"] = "roi_local_px"
    with pytest.raises(SubjectShapeCoordinatePublicationError, match="attrs differ"):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    del body.attrs["coordinate_space"]

    body["centroid_xy"].attrs["coordinate_space"] = "roi_local_px"
    with pytest.raises(SubjectShapeCoordinatePublicationError, match="attrs differ"):
        load_persisted_subject_shape_coordinate_publication(
            root,
            "analysis/subject_shape_runs/shape_001",
        )
    del body["centroid_xy"].attrs["coordinate_space"]

    # The cluster path computes and shards an explicitly unbound numeric stage.
    # Positional values are projected during chunk writing, but the projection
    # receipt is not canonical authority. Identity, descriptors, completion,
    # and selection are created only after the run is atomically renamed into
    # this authoritative archive.
    monkeypatch.setattr(
        shape_materializer,
        "write_best_effort_run_lineage_attrs",
        lambda *args, **kwargs: None,
    )

    def reject_legacy_post_write_transform(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("projected writer invoked legacy in-place transform")

    monkeypatch.setattr(
        module,
        "transform_subject_shape_geometry_to_source_camera",
        reject_legacy_post_write_transform,
    )
    scratch = tmp_path / "subject-shape-materializer"
    result = shape_materializer.materialize_subject_shape(
        tmp_path / "canonical.zarr",
        scratch_root=scratch,
        refined_run="r1",
        run_name="shape_materialized",
        block_rows=2,
        output_shard_rows=4,
        execution_backend="serial_driver",
        scheduler="single-threaded",
        num_workers=1,
        shard_copy_workers=1,
        native_threads=1,
        copy_backend="python",
        apply=True,
        keep_scratch=True,
        check_capacity=False,
        stage_command="unit-test-subject-shape-materializer",
    )
    assert result["status"] == "complete"
    scratch_compute = shape_materializer.open_zarr_root(
        scratch / "compute.zarr",
        mode="r+",
    )["analysis/subject_shape_runs/shape_materialized"]
    assert scratch_compute.attrs["coordinate_binding_status"] == (
        "unbound_numeric_stage_complete_v1"
    )
    assert scratch_compute.attrs["stage_selector_eligible"] is False
    assert "coordinate_contract" not in scratch_compute.attrs
    assert "coordinate_records" not in scratch_compute
    assert "instance_key" not in scratch_compute
    assert scratch_compute["components/subject_body"].attrs[
        "point_coordinate_space"
    ] == "source_camera_image_px_precanonical_numeric"
    scratch_manifest = scratch_compute.attrs[module.SUBJECT_SHAPE_UNBOUND_MANIFEST_ATTR]
    assert scratch_manifest["schema_version"] == 2
    assert scratch_manifest["binding_status"] == (
        module.SUBJECT_SHAPE_PROJECTED_UNBOUND_BINDING_STATUS
    )
    assert scratch_compute.attrs[module.SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR][
        "projection_role"
    ] == "private_precanonical_numeric_evidence_not_coordinate_authority"
    original_projection = copy.deepcopy(
        scratch_compute.attrs[module.SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR]
    )
    original_projection_digest = scratch_compute.attrs[
        module.SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR
    ]
    tampered_projection = copy.deepcopy(original_projection)
    tampered_projection["row_count"] = int(tampered_projection["row_count"]) + 1
    scratch_compute.attrs[module.SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR] = (
        tampered_projection
    )
    scratch_compute.attrs[module.SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR] = (
        module._canonical_sha256(tampered_projection)
    )
    with pytest.raises(
        SubjectShapeCoordinatePublicationError,
        match="differs from the freshly resolved source authority",
    ):
        subject_shape_writer.validate_unbound_subject_shape_run(
            root,
            scratch_compute,
            expected_refined_run="r1",
            expected_run_name="shape_materialized",
        )
    scratch_compute.attrs[module.SUBJECT_SHAPE_NUMERIC_PROJECTION_ATTR] = (
        original_projection
    )
    scratch_compute.attrs[module.SUBJECT_SHAPE_NUMERIC_PROJECTION_DIGEST_ATTR] = (
        original_projection_digest
    )
    assert COORDINATE_DESCRIPTOR_ATTR not in scratch_compute[
        "components/subject_body/centroid_xy"
    ].attrs
    with pytest.raises(ValueError, match="authoritative archive"):
        subject_shape_writer.bind_staged_subject_shape_run(
            root,
            scratch_compute,
            expected_refined_run="r1",
            expected_run_name="shape_materialized",
        )

    final = root["analysis/subject_shape_runs/shape_materialized"]
    assert root["analysis/subject_shape_runs"].attrs["latest"] == (
        "shape_materialized"
    )
    assert final.attrs["coordinate_binding_status"] == "bound_canonical_v2"
    assert final.attrs["stage_selector_eligible"] is True
    materialized = load_persisted_subject_shape_coordinate_publication(
        root,
        "analysis/subject_shape_runs/shape_materialized",
    )
    assert materialized.manifest.record_sha256 == final.attrs[
        "publication_manifest_sha256"
    ]
    np.testing.assert_allclose(
        final["components/subject_body/centroid_xy"][:],
        expected_centroid,
    )
    legacy = root["analysis/subject_shape_runs/shape_001"]
    legacy_arrays = dict(module._iter_arrays(legacy))
    materialized_arrays = dict(module._iter_arrays(final))
    assert set(legacy_arrays) == set(materialized_arrays)
    for path in sorted(legacy_arrays):
        np.testing.assert_array_equal(
            np.asarray(materialized_arrays[path][:]),
            np.asarray(legacy_arrays[path][:]),
            err_msg=f"fused projection changed {path}",
        )


def test_subject_shape_translation_gate_rejects_scale_or_axis_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    continuous = object()
    edge = object()
    source = SimpleNamespace(
        context=SimpleNamespace(
            row_identity=SimpleNamespace(leading_dimension=2),
            continuous_chain=continuous,
            pixel_edge_chain=edge,
        )
    )

    def scaled(values: Any, chain: Any, *, row_identity: Any) -> np.ndarray:  # noqa: ARG001
        data = np.asarray(values, dtype=np.float64)
        result = data.copy()
        result[:, 0] = 2.0 * data[:, 1] + 7.0
        result[:, 1] = data[:, 0] + 11.0
        return result

    monkeypatch.setattr(module, "apply_bound_directed_transform_chain", scaled)
    with pytest.raises(
        SubjectShapeCoordinatePublicationError,
        match="only exact translation",
    ):
        require_translation_only_refined_placement(source)


class _InterruptingAttrs(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.interrupt_eligibility_once = True

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "stage_selector_eligible" and value is True and self.interrupt_eligibility_once:
            self.interrupt_eligibility_once = False
            raise KeyboardInterrupt("simulated final eligibility interruption")
        super().__setitem__(key, value)


def test_subject_shape_activation_baseexception_restores_selectors_and_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "a" * 32
    run = SimpleNamespace(
        attrs=_InterruptingAttrs(
            {
                SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
                "stage_selector_eligible": False,
            }
        )
    )
    parent = SimpleNamespace(attrs={"latest": "old", "latest_complete": "old"})
    root = object()
    snapshot = selector_snapshot(parent)
    proof = module.BoundSubjectShapeCoordinatePublication(
        run_path="analysis/subject_shape_runs/new",
        source=object(),
        row_identity=object(),
        temporal_authority=object(),
        component_schema=object(),
        scientific_configuration=object(),
        tail_sample_axis=object(),
        derivation=object(),
        descriptors={},
        body_frame=object(),
        heading_semantics=object(),
        manifest=SimpleNamespace(record_sha256="b" * 64),
        component_names=LABELS,
        selector_eligible=False,
        publication_owner=owner,
        _root=root,
        _run=run,
        _verification_seal=module._BOUND_PUBLICATION_SEAL,
    )
    monkeypatch.setattr(
        module,
        "_node",
        lambda _root, path, **_kwargs: (
            parent if path == "analysis/subject_shape_runs" else run
        ),
    )
    monkeypatch.setattr(module, "archive_identity", lambda _value: "archive")
    monkeypatch.setattr(module, "_require_state", lambda *_args, **_kwargs: owner)
    reloads: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fresh_reload(*args: Any, **kwargs: Any) -> Any:
        reloads.append((args, kwargs))
        return proof

    monkeypatch.setattr(
        module,
        "validate_sealed_subject_shape_publication_metadata",
        fresh_reload,
    )

    with pytest.raises(KeyboardInterrupt, match="final eligibility"):
        activate_subject_shape_coordinate_publication(
            root,
            parent,
            proof,
            run_name="new",
            owner=owner,
            snapshot=snapshot,
        )

    assert run.attrs["stage_selector_eligible"] is False
    assert parent.attrs == {"latest": "old", "latest_complete": "old"}
    assert len(reloads) == 2
    assert all(
        call[1]
        == {
            "expected_selector_eligible": False,
            "expected_publication_owner": owner,
        }
        for call in reloads
    )


class _ConcurrentParentAttrs(dict[str, Any]):
    def __init__(
        self,
        *args: Any,
        trigger_key: str,
        mutation_key: str,
        mutation_value: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trigger_key = trigger_key
        self.mutation_key = mutation_key
        self.mutation_value = mutation_value
        self.triggered = False

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if key == self.trigger_key and not self.triggered:
            self.triggered = True
            super().__setitem__(self.mutation_key, copy.deepcopy(self.mutation_value))


class _TakeoverAfterEpochWriteAttrs(dict[str, Any]):
    def __init__(
        self,
        *args: Any,
        trigger_key: str,
        takeover_attrs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.trigger_key = trigger_key
        self.takeover_attrs = copy.deepcopy(takeover_attrs)
        self.triggered = False

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if key == self.trigger_key and not self.triggered:
            self.triggered = True
            for takeover_key, takeover_value in self.takeover_attrs.items():
                super().__setitem__(takeover_key, copy.deepcopy(takeover_value))
            super().pop("latest_pending", None)


class _PostWriteInterruptingAttrs(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.interrupt_eligibility_once = True

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if key == "stage_selector_eligible" and value is True and self.interrupt_eligibility_once:
            self.interrupt_eligibility_once = False
            raise KeyboardInterrupt("simulated interrupt after persisted eligibility")


class _PendingPostWriteInterruptingAttrs(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.interrupt_pending_once = True

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, value)
        if key == "latest_pending" and self.interrupt_pending_once:
            self.interrupt_pending_once = False
            raise KeyboardInterrupt("simulated interrupt after persisted pending receipt")


def _fake_subject_shape_activation_proof(
    root: Any,
    run: Any,
    *,
    owner: str,
) -> module.BoundSubjectShapeCoordinatePublication:
    return module.BoundSubjectShapeCoordinatePublication(
        run_path="analysis/subject_shape_runs/new",
        source=object(),
        row_identity=object(),
        temporal_authority=object(),
        component_schema=object(),
        scientific_configuration=object(),
        tail_sample_axis=object(),
        scalar_surfaces={},
        scalar_surface_inventory=object(),
        derivation=object(),
        descriptors={},
        body_frame=object(),
        heading_semantics=object(),
        manifest=SimpleNamespace(record_sha256="b" * 64),
        component_names=LABELS,
        selector_eligible=False,
        publication_owner=owner,
        _root=root,
        _run=run,
        _verification_seal=module._BOUND_PUBLICATION_SEAL,
    )


def _patch_fake_activation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root: Any,
    parent: Any,
    run: Any,
    proof: module.BoundSubjectShapeCoordinatePublication,
) -> None:
    monkeypatch.setattr(
        module,
        "_node",
        lambda _root, path, **_kwargs: (
            parent if path == "analysis/subject_shape_runs" else run
        ),
    )
    monkeypatch.setattr(module, "archive_identity", lambda _value: "archive")
    monkeypatch.setattr(
        module,
        "canonical_node_path",
        lambda value: (
            "analysis/subject_shape_runs"
            if value is parent
            else "analysis/subject_shape_runs/new"
            if value is run
            else ""
        ),
    )
    monkeypatch.setattr(
        module,
        "_require_state",
        lambda *_args, **_kwargs: proof.publication_owner,
    )
    monkeypatch.setattr(
        module,
        "validate_sealed_subject_shape_publication_metadata",
        lambda *_args, **_kwargs: proof,
    )
    monkeypatch.setattr(
        module,
        "_load_subject_shape_publication",
        lambda *_args, **_kwargs: pytest.fail(
            "activation used the decoded publication fallback"
        ),
    )


def test_subject_shape_activation_closes_fresh_proof_phases_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class RecordingAttrs(dict[str, Any]):
        def __init__(self, label: str, values: Mapping[str, Any]) -> None:
            super().__init__(values)
            self.label = label

        def __setitem__(self, key: str, value: Any) -> None:
            events.append(f"write:{self.label}:{key}")
            super().__setitem__(key, value)

        def __delitem__(self, key: str) -> None:
            events.append(f"delete:{self.label}:{key}")
            super().__delitem__(key)

    owner = "a" * 32
    run = SimpleNamespace(
        attrs=RecordingAttrs(
            "run",
            {
                SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
                "stage_selector_eligible": False,
            },
        )
    )
    parent = SimpleNamespace(
        attrs=RecordingAttrs(
            "parent",
            {"latest": "old", "latest_complete": "old"},
        )
    )
    root = object()
    snapshot = selector_snapshot(parent)
    proof = _fake_subject_shape_activation_proof(root, run, owner=owner)
    _patch_fake_activation(
        monkeypatch,
        root=root,
        parent=parent,
        run=run,
        proof=proof,
    )
    monkeypatch.setattr(
        module,
        "validate_sealed_subject_shape_publication_metadata",
        lambda *_args, **_kwargs: events.append("load-proof") or proof,
    )
    monkeypatch.setattr(
        module,
        "finish_proof_verification",
        lambda: events.append("finish-proof-phase"),
    )
    monkeypatch.setattr(
        module,
        "restart_proof_verification",
        lambda: events.append("restart-proof-phase"),
    )

    activate_subject_shape_coordinate_publication(
        root,
        parent,
        proof,
        run_name="new",
        owner=owner,
        snapshot=snapshot,
    )

    first_load, second_load = (
        index for index, event in enumerate(events) if event == "load-proof"
    )
    first_finish, second_finish = (
        index
        for index, event in enumerate(events)
        if event == "finish-proof-phase"
    )
    first_parent_write = next(
        index
        for index, event in enumerate(events)
        if event.startswith("write:parent:")
    )
    restart = events.index("restart-proof-phase")
    eligibility = events.index("write:run:stage_selector_eligible")

    assert first_load < first_finish < first_parent_write
    assert first_parent_write < restart < second_load < second_finish < eligibility
    assert events[-1] == "write:run:stage_selector_eligible"


def test_subject_shape_activation_rolls_back_persisted_then_interrupted_pending_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "a" * 32
    original_parent_attrs = {"latest": "old", "latest_complete": "old"}
    parent = SimpleNamespace(
        attrs=_PendingPostWriteInterruptingAttrs(original_parent_attrs)
    )
    run = SimpleNamespace(
        attrs={
            SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
            "stage_selector_eligible": False,
        }
    )
    root = object()
    snapshot = selector_snapshot(parent)
    proof = _fake_subject_shape_activation_proof(root, run, owner=owner)
    _patch_fake_activation(
        monkeypatch,
        root=root,
        parent=parent,
        run=run,
        proof=proof,
    )

    with pytest.raises(KeyboardInterrupt, match="persisted pending receipt"):
        activate_subject_shape_coordinate_publication(
            root,
            parent,
            proof,
            run_name="new",
            owner=owner,
            snapshot=snapshot,
        )

    assert dict(parent.attrs) == original_parent_attrs
    assert run.attrs["stage_selector_eligible"] is False


def test_subject_shape_activation_accepts_proven_post_write_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "a" * 32
    run = SimpleNamespace(
        attrs=_PostWriteInterruptingAttrs(
            {
                SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
                "stage_selector_eligible": False,
            }
        )
    )
    parent = SimpleNamespace(attrs={"latest": "old", "latest_complete": "old"})
    root = object()
    snapshot = selector_snapshot(parent)
    proof = _fake_subject_shape_activation_proof(root, run, owner=owner)
    _patch_fake_activation(
        monkeypatch,
        root=root,
        parent=parent,
        run=run,
        proof=proof,
    )

    activate_subject_shape_coordinate_publication(
        root,
        parent,
        proof,
        run_name="new",
        owner=owner,
        snapshot=snapshot,
    )

    assert run.attrs["stage_selector_eligible"] is True
    assert parent.attrs["latest"] == "new"
    assert parent.attrs["latest_complete"] == "new"
    assert "latest_pending" not in parent.attrs
    assert parent.attrs[module.SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR] == 1
    assert parent.attrs[module.SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR] == (
        module.SUBJECT_SHAPE_PUBLICATION_POLICY
    )
    assert parent.attrs[module.SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR][
        "publication_owner"
    ] == owner


def test_subject_shape_deferred_activation_commits_eligibility_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "a" * 32
    staging_payload = {
        "schema_id": "palette.subject_shape_run_publish.v1",
        "final_validation": {"valid": True},
    }
    run = SimpleNamespace(
        attrs={
            SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
            "cluster_output_staging": copy.deepcopy(staging_payload),
        }
    )
    parent = SimpleNamespace(attrs={"latest": "old", "latest_complete": "old"})
    root = object()
    snapshot = selector_snapshot(parent)
    proof = _fake_subject_shape_activation_proof(root, run, owner=owner)
    _patch_fake_activation(
        monkeypatch,
        root=root,
        parent=parent,
        run=run,
        proof=proof,
    )

    receipt = activate_subject_shape_coordinate_publication(
        root,
        parent,
        proof,
        run_name="new",
        owner=owner,
        snapshot=snapshot,
        defer_eligibility=True,
    )

    assert isinstance(
        receipt,
        module.DeferredSubjectShapeCoordinateActivation,
    )
    assert parent.attrs["latest"] == "new"
    assert parent.attrs["latest_complete"] == "new"
    assert run.attrs["stage_selector_eligible"] is False

    expected_run_attrs = copy.deepcopy(dict(run.attrs))
    module.commit_deferred_subject_shape_coordinate_activation(
        receipt,
        root=root,
        parent=parent,
        run=run,
        expected_run_attrs=expected_run_attrs,
    )

    assert run.attrs["stage_selector_eligible"] is True
    assert run.attrs["cluster_output_staging"] == staging_payload


def test_subject_shape_deferred_activation_rebinds_fresh_postvalidation_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "a" * 32
    stale_run = SimpleNamespace(
        attrs={
            SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
            "palette_run_completion_status": "complete",
            "stage_selector_eligible": False,
        }
    )
    parent = SimpleNamespace(attrs={"latest": "old", "latest_complete": "old"})
    root = object()
    snapshot = selector_snapshot(parent)
    proof = _fake_subject_shape_activation_proof(root, stale_run, owner=owner)
    _patch_fake_activation(
        monkeypatch,
        root=root,
        parent=parent,
        run=stale_run,
        proof=proof,
    )
    receipt = activate_subject_shape_coordinate_publication(
        root,
        parent,
        proof,
        run_name="new",
        owner=owner,
        snapshot=snapshot,
        defer_eligibility=True,
    )
    assert isinstance(receipt, module.DeferredSubjectShapeCoordinateActivation)

    staging_payload = {
        "schema_id": "palette.subject_shape_run_publish.v1",
        "final_validation": {"valid": True, "sentinel": "must-survive"},
    }
    fresh_run = SimpleNamespace(
        attrs={
            **copy.deepcopy(dict(stale_run.attrs)),
            "cluster_output_staging": copy.deepcopy(staging_payload),
            "post_validation_sentinel": "fresh-handle-only",
        }
    )
    monkeypatch.setattr(
        module,
        "_node",
        lambda _root, path, **_kwargs: (
            parent if path == "analysis/subject_shape_runs" else fresh_run
        ),
    )
    monkeypatch.setattr(
        module,
        "canonical_node_path",
        lambda value: (
            "analysis/subject_shape_runs"
            if value is parent
            else "analysis/subject_shape_runs/new"
            if value is fresh_run
            else ""
        ),
    )
    expected_run_attrs = copy.deepcopy(dict(fresh_run.attrs))

    module.commit_deferred_subject_shape_coordinate_activation(
        receipt,
        root=root,
        parent=parent,
        run=fresh_run,
        expected_run_attrs=expected_run_attrs,
    )

    assert stale_run.attrs["stage_selector_eligible"] is False
    assert fresh_run.attrs["stage_selector_eligible"] is True
    assert fresh_run.attrs["cluster_output_staging"] == staging_payload
    assert fresh_run.attrs["post_validation_sentinel"] == "fresh-handle-only"


def test_subject_shape_activation_preserves_alien_pending_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = "a" * 32
    alien = {
        "schema_id": "palette.subject_shape_publication_pending",
        "schema_version": 1,
        "policy": module.SUBJECT_SHAPE_PUBLICATION_POLICY,
        "run_path": "analysis/subject_shape_runs/alien",
        "publication_owner": "c" * 32,
        "owner_uuid": "c" * 32,
    }
    run = SimpleNamespace(
        attrs={
            SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
            "stage_selector_eligible": False,
        }
    )
    parent = SimpleNamespace(attrs={"latest": "old", "latest_complete": "old"})
    root = object()
    snapshot = selector_snapshot(parent)
    proof = _fake_subject_shape_activation_proof(root, run, owner=owner)
    _patch_fake_activation(
        monkeypatch,
        root=root,
        parent=parent,
        run=run,
        proof=proof,
    )

    def mutate_pending(*_args: Any, **_kwargs: Any) -> Any:
        parent.attrs["latest_pending"] = copy.deepcopy(alien)
        return proof

    monkeypatch.setattr(
        module,
        "validate_sealed_subject_shape_publication_metadata",
        mutate_pending,
    )
    with pytest.raises(
        SubjectShapeCoordinatePublicationError,
        match="latest_pending",
    ):
        activate_subject_shape_coordinate_publication(
            root,
            parent,
            proof,
            run_name="new",
            owner=owner,
            snapshot=snapshot,
        )

    assert parent.attrs["latest_pending"] == alien
    assert parent.attrs["latest"] == "old"
    assert parent.attrs["latest_complete"] == "old"
    assert run.attrs["stage_selector_eligible"] is False


@pytest.mark.parametrize(
    ("trigger_key", "mutation_key", "mutation_value"),
    (
        (
            module.SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR,
            module.SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR,
            99,
        ),
        (
            "latest_complete",
            module.SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR,
            "alien_policy",
        ),
        (
            "latest_complete",
            module.SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR,
            {
                "schema_id": "palette.subject_shape_publication_lease",
                "schema_version": 1,
                "policy": "alien_policy",
                "run_path": "analysis/subject_shape_runs/alien",
                "publication_owner": "c" * 32,
                "owner_uuid": "c" * 32,
            },
        ),
    ),
)
def test_subject_shape_activation_preserves_alien_lifecycle_mutation_between_writes(
    monkeypatch: pytest.MonkeyPatch,
    trigger_key: str,
    mutation_key: str,
    mutation_value: Any,
) -> None:
    owner = "a" * 32
    attrs = _ConcurrentParentAttrs(
        {"latest": "old", "latest_complete": "old"},
        trigger_key=trigger_key,
        mutation_key=mutation_key,
        mutation_value=mutation_value,
    )
    parent = SimpleNamespace(attrs=attrs)
    run = SimpleNamespace(
        attrs={
            SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
            "stage_selector_eligible": False,
        }
    )
    root = object()
    snapshot = selector_snapshot(parent)
    proof = _fake_subject_shape_activation_proof(root, run, owner=owner)
    _patch_fake_activation(
        monkeypatch,
        root=root,
        parent=parent,
        run=run,
        proof=proof,
    )

    with pytest.raises(SubjectShapeCoordinatePublicationError):
        activate_subject_shape_coordinate_publication(
            root,
            parent,
            proof,
            run_name="new",
            owner=owner,
            snapshot=snapshot,
        )

    assert parent.attrs[mutation_key] == mutation_value
    assert parent.attrs["latest"] == "old"
    assert parent.attrs["latest_complete"] == "old"
    assert "latest_pending" not in parent.attrs
    assert run.attrs["stage_selector_eligible"] is False


@pytest.mark.parametrize(
    "trigger_key",
    (
        module.SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR,
        module.SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR,
    ),
)
def test_subject_shape_activation_rollback_preserves_takeover_after_epoch_write(
    monkeypatch: pytest.MonkeyPatch,
    trigger_key: str,
) -> None:
    owner = "a" * 32
    alien_owner = "c" * 32
    alien_lease = {
        "schema_id": "palette.subject_shape_publication_lease",
        "schema_version": 1,
        "policy": module.SUBJECT_SHAPE_PUBLICATION_POLICY,
        "run_path": "analysis/subject_shape_runs/alien",
        "publication_owner": alien_owner,
        "owner_uuid": alien_owner,
        "base_generation": 0,
        "next_generation": 1,
        "pending_receipt_sha256": "d" * 64,
    }
    takeover_attrs = {
        module.SUBJECT_SHAPE_PUBLICATION_POLICY_ATTR: (
            module.SUBJECT_SHAPE_PUBLICATION_POLICY
        ),
        module.SUBJECT_SHAPE_PUBLICATION_GENERATION_ATTR: 1,
        module.SUBJECT_SHAPE_PARENT_PUBLICATION_LEASE_ATTR: alien_lease,
    }
    parent = SimpleNamespace(
        attrs=_TakeoverAfterEpochWriteAttrs(
            {"latest": "old", "latest_complete": "old"},
            trigger_key=trigger_key,
            takeover_attrs=takeover_attrs,
        )
    )
    run = SimpleNamespace(
        attrs={
            SUBJECT_SHAPE_PUBLICATION_OWNER_ATTR: owner,
            "stage_selector_eligible": False,
        }
    )
    root = object()
    snapshot = selector_snapshot(parent)
    proof = _fake_subject_shape_activation_proof(root, run, owner=owner)
    _patch_fake_activation(
        monkeypatch,
        root=root,
        parent=parent,
        run=run,
        proof=proof,
    )

    with pytest.raises(SubjectShapeCoordinatePublicationError):
        activate_subject_shape_coordinate_publication(
            root,
            parent,
            proof,
            run_name="new",
            owner=owner,
            snapshot=snapshot,
        )

    assert dict(parent.attrs) == {
        "latest": "old",
        "latest_complete": "old",
        **takeover_attrs,
    }
    assert run.attrs["stage_selector_eligible"] is False
