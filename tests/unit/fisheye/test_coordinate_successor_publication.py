from __future__ import annotations

from copy import copy, deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zarr

import pytest

from fisheye.shared.coordinate_record import (
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared import keypoint_coordinate_publication
from fisheye.shared import subject_mask_coordinate_publication
from fisheye.shared.zarr.coordinate_successor_authority import (
    COORDINATE_SUCCESSOR_AUTHORITY_ATTR,
    CoordinateSuccessorAuthorityError,
    KEYPOINT_COORDINATE_SUCCESSOR_KIND,
    build_coordinate_successor_authority,
    load_coordinate_successor_authority,
    stamp_coordinate_successor_authority,
    validate_coordinate_successor_authority,
)
from fisheye.shared.zarr.coordinate_successor_files import (
    copy_metadata_and_link_payload,
    metadata_tree_sha256,
)
from fisheye.shared.model_input_transform import ModelInputTransform
from fisheye.shared.proof_verification import proof_verification_scope
from fisheye.shared.zarr import keypoint_coordinate_successor as keypoint_successor
from fisheye.shared.zarr import (
    subject_mask_coordinate_successor as subject_mask_successor,
)
from fisheye.shared.zarr import sealed_geometry_crop_profile as sealed_crop
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropGeometryPolicy,
    CropPaddingMode,
    CropPlacementMode,
    CropSizeMode,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.pixel_frame_authority import (
    CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_ID,
    CROP_PLACEMENT_PADDED_LEGACY_PRODUCER,
    CROP_PLACEMENT_PADDED_PRODUCER,
    CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PROVENANCE_ATTR,
    array_values_sha256,
    parse_crop_placement_ownership,
)
from fisheye.shared.zarr.storage_profiles import PUBLISHED_HTTP_V1
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)
from tests.unit.fisheye.test_keypoint_coordinate_publication import (
    _artifact as _keypoint_artifact,
    _fixture as _keypoint_publication_fixture,
)
from tests.unit.fisheye.test_subject_mask_coordinate_publication import (
    LABELS as SUBJECT_MASK_LABELS,
    MODEL_ARTIFACT as SUBJECT_MASK_MODEL_ARTIFACT,
    MODEL_TRANSFORM as SUBJECT_MASK_MODEL_TRANSFORM,
    _owner as _subject_mask_owner,
    _subject_fixture_with_source,
)


class _Node:
    _archive = object()

    def __init__(self, *, path: str, children=None) -> None:
        self.path = path
        self._coordinate_archive_token = self._archive
        self.attrs: dict[str, object] = {}
        self.children = dict(children or {})

    def __getitem__(self, key: str):
        return self.children[key]


class _Array:
    def __init__(self, *, path: str, values: object) -> None:
        self.path = path
        self.attrs: dict[str, object] = {}
        self._values = np.asarray(values)
        self.shape = self._values.shape
        self.dtype = self._values.dtype

    def __getitem__(self, key: object) -> np.ndarray:
        return self._values[key]


class _Group:
    def __init__(self, *, path: str, children: dict[str, object]) -> None:
        self.path = path
        self.attrs: dict[str, object] = {
            "status": "complete",
            "stage_selector_eligible": False,
            "artifact_class": "geometry_only_analysis",
        }
        self.children = children

    def __getitem__(self, key: str) -> object:
        return self.children[key]

    def keys(self):
        return self.children.keys()


class _Root:
    def __init__(self, nodes: dict[str, object]) -> None:
        self.nodes = nodes

    def __getitem__(self, key: str) -> object:
        return self.nodes[key]


def _direct_hybrid_preprocessing() -> SimpleNamespace:
    transform = ModelInputTransform(
        name="identity",
        native_height=384,
        native_width=384,
        model_height=384,
        model_width=384,
    )
    return SimpleNamespace(
        profile_id=keypoint_successor.DIRECT_HYBRID_TERMINAL_EVIDENCE_PROFILE,
        profile_version=1,
        input_mode="numpy_list",
        document={
            "evidence_semantics": "observed_completed_inference_runtime_v1",
            "coordinate_contract_mode": "legacy_noncanonical",
            "observed_input_mode_effective": "numpy-list",
            "observed_runtime": {
                "input_mode_effective": "numpy-list",
                "model_input_transform": transform.to_attrs(),
                "model_input_shape_hw": [384, 384],
                "model_network_input_shape_hw": [384, 384],
                "native_roi_shape_hw": [384, 384],
            },
        },
    )


def test_keypoint_successor_resolves_direct_hybrid_observed_runtime() -> None:
    transform, submitted_input_mode = keypoint_successor._resolve_preprocessing_runtime(
        _direct_hybrid_preprocessing()
    )

    assert (
        transform.to_attrs()
        == ModelInputTransform(
            name="identity",
            native_height=384,
            native_width=384,
            model_height=384,
            model_width=384,
        ).to_attrs()
    )
    assert submitted_input_mode == "numpy-list"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_input_mode_effective", "tensor"),
        ("coordinate_contract_mode", "canonical_v2"),
    ],
)
def test_keypoint_successor_rejects_inconsistent_direct_hybrid_profile(
    field: str,
    value: str,
) -> None:
    preprocessing = _direct_hybrid_preprocessing()
    preprocessing.document[field] = value

    with pytest.raises(ValueError, match="runtime evidence is inconsistent"):
        keypoint_successor._resolve_preprocessing_runtime(preprocessing)


def test_keypoint_successor_rejects_direct_hybrid_runtime_extent_mismatch() -> None:
    preprocessing = _direct_hybrid_preprocessing()
    preprocessing.document["observed_runtime"]["native_roi_shape_hw"] = [512, 512]

    with pytest.raises(ValueError, match="runtime extents differ"):
        keypoint_successor._resolve_preprocessing_runtime(preprocessing)


@pytest.mark.parametrize(
    ("probability_dtype", "materialize_binary", "expected_encoding"),
    [
        (np.uint8, False, "linear_uint8_0_255"),
        (np.float16, True, "unit_float"),
    ],
)
def test_subject_mask_successor_derives_semantics_from_schema_and_arrays(
    tmp_path: Path,
    probability_dtype: type[np.generic],
    materialize_binary: bool,
    expected_encoding: str,
) -> None:
    root = zarr.open_group(str(tmp_path / "semantic.zarr"), mode="w")
    run = root.create_group("subject_mask_runs/source")
    run.create_array(
        "mask_probs_roi",
        data=np.zeros((2, 3, 4, 5), dtype=probability_dtype),
    )
    if materialize_binary:
        run.create_array(
            "masks_roi",
            data=np.zeros((2, 3, 4, 5), dtype=np.uint8),
        )
    manifest = {
        "payload": {
            "logical_schema": {
                "schema_id": "palette.stage.subject_mask_probabilities_test",
                "schema_version": 1,
                "components": {
                    "labels": ["subject_body", "eyes_union", "swim_bladder"]
                },
                "output_semantics": "multilabel",
                "overlap_policy": "independent_sigmoid",
                "probability_semantics": "sigmoid_multilabel_logits",
                "probability_encoding": expected_encoding,
                "threshold": 0.5,
            }
        }
    }

    record = subject_mask_successor._raw_semantic_normalization(
        run,
        manifest=manifest,
        threshold=0.5,
    )

    attrs = record["resolved_run_attrs"]
    assert attrs["probabilities_dtype"] == np.dtype(probability_dtype).name
    assert attrs["probabilities_encoding"] == expected_encoding
    assert attrs["masks_roi_materialized"] is materialize_binary
    assert attrs["binary_masks_materialized"] is materialize_binary
    assert attrs["binary_masks_source"] == (
        "threshold(mask_probs_roi, threshold=0.5)"
        if materialize_binary
        else "not_materialized"
    )


def test_subject_mask_successor_rejects_schema_dtype_encoding_disagreement(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "semantic-mismatch.zarr"), mode="w")
    run = root.create_group("subject_mask_runs/source")
    run.create_array("mask_probs_roi", data=np.zeros((1, 1, 2, 2), dtype=np.uint8))
    manifest = {
        "payload": {
            "logical_schema": {
                "schema_id": "palette.stage.subject_mask_probabilities_test",
                "schema_version": 1,
                "components": {"labels": ["subject_body"]},
                "output_semantics": "multilabel",
                "overlap_policy": "independent_sigmoid",
                "probability_semantics": "sigmoid_multilabel_logits",
                "probability_encoding": "unit_float",
                "threshold": 0.5,
            }
        }
    }

    with pytest.raises(ValueError, match="probability encoding differs"):
        subject_mask_successor._raw_semantic_normalization(
            run,
            manifest=manifest,
            threshold=0.5,
        )


def test_subject_mask_successor_binds_refined_semantics_to_core_source() -> None:
    evidence_path = "refined_subject_masks_runs/refined_worker_draft"
    evidence = _Node(path=evidence_path)
    evidence.attrs.update(
        {
            subject_mask_successor.RUN_COMPLETION_STATUS_ATTR: "complete",
            "stage_selector_eligible": False,
            "derived_mask_caches_stale": False,
            "metrics_stale": False,
            "contours_stale": False,
            "masks_roi_materialized": True,
            "mask_storage_authority": "masks_roi",
            "editable_mask_surface": "masks_roi",
            "mask_bitpacked_materialized": False,
            "mask_rle_materialized": False,
            "bbox_xyxy_convention": "pixel_edge_half_open",
            "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
        }
    )
    manifest = {
        "payload_digest": "a" * 64,
        "payload": {
            "source": {
                "run_path": evidence_path,
                "validation_receipt": {
                    "schema_id": "palette.subject_mask.source_validation_receipt",
                    "schema_version": 2,
                    "payload_digest": "b" * 64,
                },
            }
        },
    }

    record = subject_mask_successor._refined_semantic_normalization(
        evidence,
        source_manifest=manifest,
        evidence_run_path=evidence_path,
    )

    assert record["resolved_run_attrs"]["derived_mask_caches_stale"] is False
    assert record["resolved_run_attrs"]["metrics_stale"] is False
    assert record["resolved_run_attrs"]["contours_stale"] is False
    assert record["resolved_run_attrs"]["bbox_xyxy_convention"] == (
        "pixel_edge_half_open"
    )
    assert record["resolved_run_attrs"]["bbox_xyxy_derivation"] == (
        "foreground_half_open_pixel_edges_xyxy_v1"
    )
    assert record["evidence"]["worker_draft_run_path"] == evidence_path


@pytest.mark.parametrize(
    ("attr_name", "replacement"),
    [
        ("derived_mask_caches_stale", True),
        ("metrics_stale", 0),
        ("mask_storage_authority", "legacy_cache"),
        ("bbox_xyxy_convention", "pixel_center_inclusive"),
        ("bbox_xyxy_derivation", "legacy_unspecified"),
    ],
)
def test_subject_mask_successor_rejects_inexact_refined_semantics(
    attr_name: str,
    replacement: object,
) -> None:
    evidence_path = "refined_subject_masks_runs/refined_worker_draft"
    evidence = _Node(path=evidence_path)
    evidence.attrs.update(
        {
            subject_mask_successor.RUN_COMPLETION_STATUS_ATTR: "complete",
            "stage_selector_eligible": False,
            "derived_mask_caches_stale": False,
            "metrics_stale": False,
            "contours_stale": False,
            "masks_roi_materialized": True,
            "mask_storage_authority": "masks_roi",
            "editable_mask_surface": "masks_roi",
            "mask_bitpacked_materialized": False,
            "mask_rle_materialized": False,
            "bbox_xyxy_convention": "pixel_edge_half_open",
            "bbox_xyxy_derivation": "foreground_half_open_pixel_edges_xyxy_v1",
        }
    )
    evidence.attrs[attr_name] = replacement
    manifest = {
        "payload_digest": "a" * 64,
        "payload": {
            "source": {
                "run_path": evidence_path,
                "validation_receipt": {
                    "schema_id": "palette.subject_mask.source_validation_receipt",
                    "schema_version": 2,
                    "payload_digest": "b" * 64,
                },
            }
        },
    }

    with pytest.raises(ValueError, match="semantic evidence"):
        subject_mask_successor._refined_semantic_normalization(
            evidence,
            source_manifest=manifest,
            evidence_run_path=evidence_path,
        )


def test_subject_mask_successor_record_pointers_accept_bound_and_persisted_records() -> (
    None
):
    pointers = subject_mask_successor._record_pointers(
        {
            "bound": SimpleNamespace(
                record_ref="/subject_mask_runs/run@bound",
                record_sha256="a" * 64,
            ),
            "persisted": {
                "record_ref": "/subject_mask_runs/run@persisted",
                "record_sha256": "b" * 64,
            },
        }
    )

    assert pointers == {
        "bound": {
            "record_ref": "/subject_mask_runs/run@bound",
            "record_sha256": "a" * 64,
        },
        "persisted": {
            "record_ref": "/subject_mask_runs/run@persisted",
            "record_sha256": "b" * 64,
        },
    }


def test_subject_mask_successor_stamps_coordinate_receipt_from_closed_evidence() -> None:
    run = _Node(path="subject_mask_runs/successor", children={})
    names = subject_mask_successor._RAW_COORDINATE_VALIDATION_RECORD_NAMES
    coordinate_records = {
        name: {
            "record_ref": f"/subject_mask_runs/successor@{name}",
            "record_sha256": f"{index + 1:064x}",
        }
        for index, name in enumerate(names)
    }
    source_manifest = {
        "payload_digest": "a" * 64,
        "payload": {
            "logical_content": {"digest": "b" * 64},
            "source": {
                "validation_receipt": {
                    "schema_id": "palette.subject_mask.source_validation_receipt",
                    "schema_version": 2,
                    "payload_digest": "c" * 64,
                    "document_sha256": "d" * 64,
                    "semantic_unit_count": 4,
                }
            },
        },
    }
    pointer = subject_mask_successor._stamp_coordinate_validation_receipt(
        run,
        kind="raw_subject_mask",
        successor_run_path="subject_mask_runs/successor",
        source_manifest=source_manifest,
        source_run_path="subject_mask_runs/source",
        bundle_authority_kind="inactive_subject_mask_bundle_v3",
        bundle_manifest={"bundle": "authority"},
        coordinate_records=coordinate_records,
        coordinate_record_names=names,
        payload_equivalence={
            "schema_id": "palette.coordinate_successor_payload_file_equivalence",
            "schema_version": 1,
            "receipt_digest": "e" * 64,
            "inventory_digest": "f" * 64,
            "payload_file_count": 3,
        },
    )

    assert pointer is not None
    assert pointer["record_ref"] == (
        "/subject_mask_runs/successor@coordinate_surface_validation_receipt"
    )
    receipt = run.attrs["coordinate_surface_validation_receipt"]
    assert receipt["payload"]["coordinate_records"] == coordinate_records
    assert receipt["payload"]["source"]["logical_content_digest"] == "b" * 64


def test_subject_mask_successor_omits_reuse_receipt_without_source_receipt() -> None:
    run = _Node(path="subject_mask_runs/successor", children={})
    observed = subject_mask_successor._stamp_coordinate_validation_receipt(
        run,
        kind="raw_subject_mask",
        successor_run_path="subject_mask_runs/successor",
        source_manifest={
            "payload_digest": "a" * 64,
            "payload": {
                "logical_content": {"digest": "b" * 64},
                "source": {"validation_receipt": None},
            },
        },
        source_run_path="subject_mask_runs/source",
        bundle_authority_kind="inactive_subject_mask_bundle_v3",
        bundle_manifest={"bundle": "authority"},
        coordinate_records={
            "context": {
                "record_ref": "/subject_mask_runs/successor@context",
                "record_sha256": "c" * 64,
            }
        },
        coordinate_record_names=("context",),
        payload_equivalence={
            "schema_id": "palette.coordinate_successor_payload_file_equivalence",
            "schema_version": 1,
            "receipt_digest": "d" * 64,
            "inventory_digest": "e" * 64,
            "payload_file_count": 1,
        },
    )

    assert observed is None
    assert "coordinate_surface_validation_receipt" not in run.attrs


def test_subject_mask_successor_fresh_archive_child_retains_record_path(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "fresh-child.zarr"
    root = zarr.open_group(str(archive), mode="w")
    run = root.create_group("subject_mask_runs/successor")
    record = {
        "schema_id": "palette.test.coordinate_record",
        "schema_version": 1,
    }
    run.attrs["test_record"] = record
    run.attrs["test_record_sha256"] = canonical_json_sha256(record)

    fresh_root = zarr.open_group(str(archive), mode="a", use_consolidated=False)
    fresh_run = fresh_root["subject_mask_runs/successor"]

    assert sealed_crop.bind_persisted_run_attribute_record(
        fresh_run, attr_name="test_record"
    ) == {
        "record_ref": "/subject_mask_runs/successor@test_record",
        "record_sha256": canonical_json_sha256(record),
    }


def test_subject_mask_successor_uses_shared_resolver_for_refined_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    expected_surfaces = object()
    expected_root = {"refined_subject_masks_runs/refined": "refined-run"}

    def fake_prepare(root, run_path, **kwargs):
        assert root is expected_root
        assert run_path == "refined_subject_masks_runs/refined"
        assert kwargs["source_subject_mask_path"] == "subject_mask_runs/raw"
        assert kwargs["source_selector_eligible"] is False
        calls.append("prepare")

    def fake_stamp(run, binding):
        assert run == "refined-run"
        assert binding == "sealed-crop"
        calls.append("stamp")

    def fake_publish(root, run_path, **kwargs):
        assert root is expected_root
        assert run_path == "refined_subject_masks_runs/refined"
        assert kwargs["expected_publication_owner"] == "owner"
        calls.append("publish")
        return expected_surfaces

    monkeypatch.setattr(
        subject_mask_successor,
        "prepare_refined_subject_mask_coordinate_context",
        fake_prepare,
    )
    monkeypatch.setattr(
        subject_mask_successor,
        "stamp_persisted_padded_placement_provenance",
        fake_stamp,
    )
    monkeypatch.setattr(
        subject_mask_successor,
        "publish_refined_subject_mask_coordinate_surfaces",
        fake_publish,
    )

    observed = subject_mask_successor._publish_refined_with_sealed_crop(
        expected_root,
        refined_run_path="refined_subject_masks_runs/refined",
        refined_owner="owner",
        raw_run_path="subject_mask_runs/raw",
        mask_labels=["subject_body"],
        sealed_crop="sealed-crop",
    )

    assert observed is expected_surfaces
    assert calls == ["stamp", "prepare", "publish"]


def _sealed_crop_fixture(tmp_path: Path):
    n_frames = 5
    n_instances = 4
    width = height = 100
    row_signature = np.arange(n_instances * 32, dtype=np.uint8).reshape(n_instances, 32)
    crop_arrays = {
        name: _Array(path=f"crop_runs/crop/{name}", values=np.zeros(1, dtype=np.uint8))
        for name in CROP_GEOMETRY_SCHEMA_V1.binding_paths
    }
    crop_arrays["instance_key"] = _Array(
        path="crop_runs/crop/instance_key",
        values=np.asarray([10, 11, 12, 13], dtype="<u8"),
    )
    crop_arrays["source_acquisition_frame_index"] = _Array(
        path="crop_runs/crop/source_acquisition_frame_index",
        values=np.asarray([0, 1, 3, 4], dtype="<i8"),
    )
    crop_arrays["source_row_signature"] = _Array(
        path="crop_runs/crop/source_row_signature", values=row_signature
    )
    crop_arrays["source_crop_xywh"] = _Array(
        path="crop_runs/crop/source_crop_xywh",
        values=np.asarray(
            [[0, 0, 384, 384], [1, 2, 384, 384], [3, 4, 384, 384], [5, 6, 384, 384]],
            dtype="<f4",
        ),
    )
    crop_arrays["roi_coordinates_full"] = _Array(
        path="crop_runs/crop/roi_coordinates_full",
        values=np.asarray([[0, 0], [1, 2], [3, 4], [5, 6]], dtype="<i4"),
    )
    crop_arrays["roi_sizes_full"] = _Array(
        path="crop_runs/crop/roi_sizes_full",
        values=np.full((n_instances, 2), 384, dtype="<i4"),
    )
    crop_group = _Group(
        path="crop_runs/crop",
        children=dict(crop_arrays),
    )
    logical = {
        "digest_algorithm": "sha256_canonical_json_v1",
        "digest": "b" * 64,
        "document": {"arrays": {"source_row_signature": {"sha256": "c" * 64}}},
    }
    crop_policy = CropGeometryPolicy(
        purpose="subject_pose",
        size_mode=CropSizeMode.FIXED_PER_RUN,
        fixed_size_wh=(384, 384),
        padding_mode=CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME,
        placement_mode=CropPlacementMode.VERIFIED_EXPLICIT_PER_ROW,
        placement_authority={
            "schema_id": "palette.crop_geometry.explicit_origin_authority",
            "schema_version": 1,
            "authority_kind": "signed_hybrid_crop_provider",
            "run_id": "crop",
            "provider_record_sha256": "e" * 64,
            "source_rowset_fingerprint": "f" * 64,
            "source_pixel_fingerprint": "1" * 64,
            "source_row_signature_spec_digest": "2" * 64,
        },
    )
    logical_schema = {
        "dimensions": {
            "n_frames": n_frames,
            "n_instances": n_instances,
            "source_width": width,
            "source_height": height,
        },
        "crop_policy": crop_policy.as_manifest(),
    }
    manifest = {
        "schema_id": "palette.crop_geometry.run_manifest",
        "schema_version": 2,
        "payload_digest": "a" * 64,
        "payload": {
            "run_id": "crop",
            "logical_schema": logical_schema,
            "logical_content": logical,
            "coordinate_contract": {"digest": "d" * 64},
            "source_pixel_authority": {
                "n_frames": n_frames,
                "source_width": width,
                "source_height": height,
                "camera_identity": "cam",
            },
        },
    }
    crop_group.attrs["run_manifest"] = manifest
    crop_reference = {
        "run_path": "crop_runs/crop",
        "manifest_digest": canonical_json_sha256(manifest),
        "manifest_payload_digest": manifest["payload_digest"],
        "logical_content_digest": logical["digest"],
        "row_signatures_digest": "c" * 64,
        "coordinate_catalog_digest": "d" * 64,
    }
    source_arrays = {
        "source_crop_row_ids": _Array(
            path="source/source_crop_row_ids",
            values=np.arange(n_instances, dtype="<i8"),
        ),
        "instance_key": crop_arrays["instance_key"],
        "source_acquisition_frame_index": crop_arrays["source_acquisition_frame_index"],
        "source_crop_row_signature": crop_arrays["source_row_signature"],
    }
    acquisition = SimpleNamespace(record=SimpleNamespace(camera_id="cam"))
    point = SimpleNamespace(
        pixel_convention="continuous",
        endpoint=SimpleNamespace(width=width, height=height),
        record_ref="/camera/continuous",
        record_sha256="e" * 64,
    )
    bbox = SimpleNamespace(
        pixel_convention="pixel_edge_half_open",
        endpoint=SimpleNamespace(width=width, height=height),
        record_ref="/camera/pixel_edge_half_open",
        record_sha256="f" * 64,
    )
    root = _Root(
        {
            "crop_runs/crop": crop_group,
            "analysis/coordinate_frames/source_camera/cam/continuous": point,
            "analysis/coordinate_frames/source_camera/cam/pixel_edge_half_open": bbox,
        }
    )
    source_manifest = {
        "payload": {
            "logical_schema": {
                "dimensions": {"n_frames": n_frames, "n_instances": n_instances}
            }
        }
    }
    transform = ModelInputTransform(
        name="identity",
        native_height=384,
        native_width=384,
        model_height=384,
        model_width=384,
    )
    return {
        "archive": tmp_path,
        "root": root,
        "crop_group": crop_group,
        "crop_arrays": crop_arrays,
        "manifest": manifest,
        "crop_reference": crop_reference,
        "source_manifest": source_manifest,
        "source_arrays": source_arrays,
        "acquisition": acquisition,
        "point": point,
        "bbox": bbox,
        "transform": transform,
    }


def _patch_sealed_crop_fixture(monkeypatch, fixture) -> None:
    monkeypatch.setattr(
        sealed_crop,
        "open_persisted_crop_geometry_publication",
        lambda archive, run_id: SimpleNamespace(
            manifest=fixture["manifest"], arrays=fixture["crop_arrays"]
        ),
    )
    monkeypatch.setattr(
        sealed_crop, "validate_crop_run_manifest", lambda manifest: ()
    )
    monkeypatch.setattr(
        sealed_crop,
        "load_persisted_acquisition_camera_authority",
        lambda root: (None, fixture["acquisition"]),
    )
    monkeypatch.setattr(
        sealed_crop,
        "load_source_camera_pixel_frame_authority",
        lambda node, *, acquisition_frame: node,
    )


def _source_manifest() -> dict[str, object]:
    return {
        "schema_id": "palette.keypoint.run_manifest",
        "schema_version": 1,
        "payload_digest": "1" * 64,
        "payload": {"logical_content": {"digest": "2" * 64}},
    }


def _authority(run: _Node) -> dict[str, object]:
    coordinate = stamp_and_bind_persisted_coordinate_record(
        run,
        {"schema_id": "coordinate-test", "schema_version": 1},
        attr_name="coordinate_context",
    )
    return build_coordinate_successor_authority(
        kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
        source_family="keypoints_runs",
        source_run_path="keypoints_runs/source",
        source_manifest=_source_manifest(),
        source_authority_kind="sealed_keypoint_bundle",
        source_authority={"schema_id": "source-authority", "schema_version": 1},
        successor_family="keypoints_runs",
        successor_run_path=run.path,
        payload_equivalence={"mode": "hardlink_exact_payload"},
        coordinate_records={
            "context": {
                "record_ref": coordinate.record_ref,
                "record_sha256": coordinate.record_sha256,
            }
        },
    )


def test_coordinate_successor_authority_revalidates_persisted_record() -> None:
    run = _Node(path="keypoints_runs/successor")
    authority = _authority(run)

    assert validate_coordinate_successor_authority(authority) == ()
    stamp_coordinate_successor_authority(run, authority)
    assert (
        load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run.path,
        )
        == authority
    )

    run.attrs["coordinate_context"] = {
        "schema_id": "coordinate-test",
        "schema_version": 2,
    }
    with pytest.raises(
        CoordinateSuccessorAuthorityError,
        match="missing, malformed, or stale",
    ):
        load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run.path,
        )


def test_coordinate_successor_authority_tampering_fails_closed() -> None:
    run = _Node(path="keypoints_runs/successor")
    authority = _authority(run)
    changed = deepcopy(authority)
    changed["payload"]["successor"]["stage_selector_eligible"] = True

    errors = validate_coordinate_successor_authority(changed)

    assert "coordinate successor authority payload digest differs" in errors
    assert "coordinate successor target lifecycle is invalid" in errors

    stamp_coordinate_successor_authority(run, authority)
    persisted = deepcopy(run.attrs[COORDINATE_SUCCESSOR_AUTHORITY_ATTR])
    persisted["payload"]["production_state_changes"] = ["latest"]
    run.attrs[COORDINATE_SUCCESSOR_AUTHORITY_ATTR] = persisted
    with pytest.raises(CoordinateSuccessorAuthorityError, match="payload digest"):
        load_coordinate_successor_authority(
            run,
            expected_kind=KEYPOINT_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run.path,
        )


def test_successor_copy_separates_metadata_and_hardlinks_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "keypoints" / "c").mkdir(parents=True)
    (source / "zarr.json").write_text('{"node_type":"group"}\n')
    (source / "keypoints" / "zarr.json").write_text('{"node_type":"array"}\n')
    (source / "keypoints" / "c" / "0").write_bytes(b"payload")
    source_metadata = metadata_tree_sha256(source)

    receipt = copy_metadata_and_link_payload(source, target)

    assert receipt == {
        "metadata_files_copied": 2,
        "payload_files_hardlinked": 1,
    }
    assert metadata_tree_sha256(source) == source_metadata
    assert os.stat(source / "zarr.json").st_ino != os.stat(target / "zarr.json").st_ino
    assert (
        os.stat(source / "keypoints" / "c" / "0").st_ino
        == os.stat(target / "keypoints" / "c" / "0").st_ino
    )

    (target / "zarr.json").write_text('{"node_type":"group","attributes":{}}\n')
    assert metadata_tree_sha256(source) == source_metadata
    assert (source / "keypoints" / "c" / "0").read_bytes() == b"payload"


def test_sealed_geometry_crop_profile_proves_keypoint_source_and_records_no_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _sealed_crop_fixture(tmp_path)
    fixture["root"]._coordinate_archive_token = object()
    _patch_sealed_crop_fixture(monkeypatch, fixture)

    binding = sealed_crop.bind_sealed_geometry_crop_successor_source(
        analysis_zarr=fixture["archive"],
        root=fixture["root"],
        crop_reference=fixture["crop_reference"],
        source_manifest=fixture["source_manifest"],
        source_arrays=fixture["source_arrays"],
        source_run_path="keypoints_runs/source",
        model_input_transform=fixture["transform"],
    )

    record = binding.as_record()
    assert record["adapter_kind"] == (
        "sealed_geometry_only_crop_manifest_v2_no_pixel_source"
    )
    assert record["crop_path"] == "crop_runs/crop"
    assert record["crop_manifest_digest"] == canonical_json_sha256(fixture["manifest"])
    assert record["model_input_transform"] == fixture["transform"].to_attrs()
    assert record["pixel_source"] == "none_historical_coordinate_migration_only"
    assert record["ephemeral_flat_pixel_cache"] == "not_an_immutable_source"
    assert "coordinate_contract" not in fixture["crop_group"].attrs
    assert fixture["crop_group"].attrs["stage_selector_eligible"] is False
    durable_ref = "/crop_runs/crop@run_manifest"
    durable_digest = canonical_json_sha256(fixture["crop_group"].attrs["run_manifest"])
    evidence = binding.source.crop_geometry.source_geometry.frame_evidence
    refs = [
        binding.source.crop_geometry.selection_derivation,
        binding.source.crop_geometry.row_identity,
        binding.source.roi_geometry.derivation,
        binding.source.roi_frame,
        binding.source.bbox_roi_frame,
        evidence.normalized_frame,
        *evidence.normalized_to_source_camera.transform_records,
    ]
    assert refs
    assert all(item.record_ref == durable_ref for item in refs)
    assert all(item.record_sha256 == durable_digest for item in refs)
    for item in refs:
        record_path, attr_name = item.record_ref[1:].split("@", 1)
        persisted = fixture["root"][record_path].attrs[attr_name]
        assert canonical_json_sha256(persisted) == item.record_sha256
    assert record["padded_crop_lineage"]["padded_row_count"] == 4
    assert record["padded_crop_lineage"]["max_padding_ltrb"] == [0, 0, 289, 290]


def test_sealed_geometry_profile_is_shared_by_keypoint_and_mask_resolvers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _sealed_crop_fixture(tmp_path)
    fixture["root"]._coordinate_archive_token = object()
    _patch_sealed_crop_fixture(monkeypatch, fixture)
    binding = sealed_crop.bind_sealed_geometry_crop_successor_source(
        analysis_zarr=fixture["archive"],
        root=fixture["root"],
        crop_reference=fixture["crop_reference"],
        source_manifest=fixture["source_manifest"],
        source_arrays=fixture["source_arrays"],
        source_run_path="keypoints_runs/source",
        model_input_transform=fixture["transform"],
    )
    from fisheye.shared import keypoint_coordinate_publication as keypoint_publication
    from fisheye.shared import subject_mask_coordinate_publication as mask_publication

    original_keypoint_loader = keypoint_publication.load_persisted_keypoint_crop_source
    original_mask_loader = mask_publication.load_persisted_subject_mask_crop_source
    monkeypatch.setattr(
        sealed_crop,
        "load_sealed_geometry_crop_source",
        lambda root, path: binding.source,
    )
    with proof_verification_scope():
        assert (
            keypoint_publication.load_persisted_keypoint_crop_source(
                fixture["root"], binding.crop_path
            )
            is binding.source
        )
        assert (
            mask_publication.load_persisted_subject_mask_crop_source(
                fixture["root"], binding.crop_path
            )
            is binding.source
        )
        assert (
            binding.source.placement_ownership_attr,
            binding.source.placement_pixel_center_ownership_attr,
            binding.source.placement_pixel_edge_ownership_attr,
        ) == (
            CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
            CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
            CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
        )
    assert not hasattr(sealed_crop, "historical_geometry_only_crop_loader")
    assert (
        keypoint_publication.load_persisted_keypoint_crop_source
        is original_keypoint_loader
    )
    assert (
        mask_publication.load_persisted_subject_mask_crop_source is original_mask_loader
    )


def test_shared_keypoint_crop_loader_dispatches_geometry_only_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _sealed_crop_fixture(tmp_path)
    from fisheye.shared import keypoint_coordinate_publication as keypoint_publication

    expected = object()
    monkeypatch.setattr(
        sealed_crop,
        "load_sealed_geometry_crop_source",
        lambda root, path: expected,
    )

    assert (
        keypoint_publication._load_persisted_keypoint_crop_source_fresh(
            fixture["root"], "crop_runs/crop"
        )
        is expected
    )


@pytest.mark.parametrize(
    "mutation", ["manifest_digest", "extent", "missing_array", "model"]
)
def test_sealed_geometry_crop_profile_rejects_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    fixture = _sealed_crop_fixture(tmp_path)
    _patch_sealed_crop_fixture(monkeypatch, fixture)
    crop_reference = dict(fixture["crop_reference"])
    transform = fixture["transform"]
    if mutation == "manifest_digest":
        crop_reference["manifest_digest"] = "9" * 64
    elif mutation == "extent":
        fixture["point"].endpoint.width = 101
    elif mutation == "missing_array":
        fixture["crop_arrays"] = dict(fixture["crop_arrays"])
        del fixture["crop_arrays"]["source_crop_xywh"]
    elif mutation == "model":
        transform = ModelInputTransform(
            name="identity",
            native_height=385,
            native_width=384,
            model_height=385,
            model_width=384,
        )

    with pytest.raises(sealed_crop.SealedGeometryCropProfileError):
        sealed_crop.bind_sealed_geometry_crop_successor_source(
            analysis_zarr=fixture["archive"],
            root=fixture["root"],
            crop_reference=crop_reference,
            source_manifest=fixture["source_manifest"],
            source_arrays=fixture["source_arrays"],
            source_run_path="keypoints_runs/source",
            model_input_transform=transform,
        )


def test_sealed_geometry_crop_profile_rejects_raw_mask_rowset_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _sealed_crop_fixture(tmp_path)
    _patch_sealed_crop_fixture(monkeypatch, fixture)
    source_arrays = {
        "source_crop_row_ids": fixture["source_arrays"]["source_crop_row_ids"],
        "instance_key": fixture["source_arrays"]["instance_key"],
        "source_acquisition_frame_index": fixture["source_arrays"][
            "source_acquisition_frame_index"
        ],
        "source_crop_xywh": _Array(
            path="source/source_crop_xywh",
            values=np.asarray(fixture["crop_arrays"]["source_crop_xywh"][...] + 1),
        ),
    }
    with pytest.raises(
        sealed_crop.SealedGeometryCropProfileError,
        match="placement",
    ):
        sealed_crop.bind_sealed_geometry_crop_successor_source(
            analysis_zarr=fixture["archive"],
            root=fixture["root"],
            crop_reference=fixture["crop_reference"],
            source_manifest=fixture["source_manifest"],
            source_arrays=source_arrays,
            source_run_path="subject_mask_runs/raw",
            model_input_transform=fixture["transform"],
        )


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ((-32, 100), [32, 0, 0, 0]),
        ((100, -19), [0, 19, 0, 0]),
        ((4204, 100), [0, 0, 76, 0]),
        ((100, 4242), [0, 0, 0, 114]),
        ((100, 100), [0, 0, 0, 0]),
    ],
)
def test_sealed_geometry_padded_crop_lineage_freezes_each_boundary(
    origin: tuple[int, int], expected: list[int]
) -> None:
    values = np.asarray([[origin[0], origin[1], 384, 384]], dtype="<f4")
    record = sealed_crop._padded_lineage_summary(
        values,
        source_width=4512,
        source_height=4512,
        roi_width=384,
        roi_height=384,
        crop_policy_payload_digest="a" * 64,
        origin_authority_digest="b" * 64,
        provider_record_sha256="c" * 64,
    )
    assert record["max_padding_ltrb"] == expected
    assert record["padded_row_count"] == int(any(expected))
    assert record["crop_local_to_source_camera"]["source_pixels_outside_extent"] == (
        "synthetic_zero_padding_no_source_pixel_correspondence"
    )


def test_sealed_geometry_padded_crop_lineage_freezes_exact_clipping_and_transform() -> None:
    values = np.asarray([[-32, -19, 384, 384]], dtype="<f4")
    record = sealed_crop._padded_lineage_summary(
        values,
        source_width=4512,
        source_height=4512,
        roi_width=384,
        roi_height=384,
        crop_policy_payload_digest="a" * 64,
        origin_authority_digest="b" * 64,
        provider_record_sha256="c" * 64,
    )
    clipped = np.asarray([[0, 0, 352, 365]], dtype="<i8")
    padding = np.asarray([[32, 19, 0, 0]], dtype="<i8")
    assert record["clipped_source_camera_intersection"][
        "sha256"
    ] == array_values_sha256(clipped)
    assert record["zero_padding_offsets_ltrb"]["sha256"] == array_values_sha256(padding)
    assert record["requested_roi"]["extent"] == {
        "width": 384,
        "height": 384,
        "units": "px",
    }


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[0.5, 0, 384, 384]], dtype="<f4"),
        np.asarray([[0, 0, 385, 384]], dtype="<f4"),
        np.asarray([[-400, 0, 384, 384]], dtype="<f4"),
        np.asarray([[0, 0, np.nan, 384]], dtype="<f4"),
    ],
)
def test_sealed_geometry_padded_crop_lineage_rejects_malformed_or_excess_windows(
    values: np.ndarray,
) -> None:
    with pytest.raises(sealed_crop.SealedGeometryCropProfileError):
        sealed_crop._padded_lineage_summary(
            values,
            source_width=4512,
            source_height=4512,
            roi_width=384,
            roi_height=384,
            crop_policy_payload_digest="a" * 64,
            origin_authority_digest="b" * 64,
            provider_record_sha256="c" * 64,
        )


def test_keypoint_auxiliary_plan_materializes_missing_arrays_without_overwriting_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _sealed_crop_fixture(tmp_path)
    _patch_sealed_crop_fixture(monkeypatch, fixture)
    source_manifest = deepcopy(fixture["source_manifest"])
    source_manifest["payload"]["storage_plan"] = {
        "storage_profile": PUBLISHED_HTTP_V1.as_manifest()
    }
    binding = sealed_crop.bind_sealed_geometry_crop_successor_source(
        analysis_zarr=fixture["archive"],
        root=fixture["root"],
        crop_reference=fixture["crop_reference"],
        source_manifest=source_manifest,
        source_arrays=fixture["source_arrays"],
        source_run_path="keypoints_runs/source",
        model_input_transform=fixture["transform"],
    )
    origins = np.asarray([[0, 0], [1, 2], [3, 4], [5, 6]], dtype="<f4")
    keypoints_roi = np.asarray(
        [
            [[1, 2], [np.nan, np.nan]],
            [[3, 4], [5, 6]],
            [[7, 8], [9, 10]],
            [[11, 12], [13, 14]],
        ],
        dtype="<f4",
    )
    bbox_roi = np.asarray(
        [[1, 2, 10, 20], [2, 3, 11, 21], [3, 4, 12, 22], [4, 5, 13, 23]],
        dtype="<f4",
    )
    keypoints_img = keypoints_roi + origins[:, None, :]
    keypoints_img[~np.isfinite(keypoints_roi)] = np.nan
    bbox_img = bbox_roi + np.column_stack((origins, origins))
    source = _Group(
        path="keypoints_runs/source",
        children={
            "source_crop_row_ids": _Array(
                path="keypoints_runs/source/source_crop_row_ids",
                values=np.arange(4, dtype="<i8"),
            ),
            "keypoints_roi": _Array(
                path="keypoints_runs/source/keypoints_roi", values=keypoints_roi
            ),
            "keypoints_img": _Array(
                path="keypoints_runs/source/keypoints_img", values=keypoints_img
            ),
            "pose_bbox_xyxy_roi": _Array(
                path="keypoints_runs/source/pose_bbox_xyxy_roi", values=bbox_roi
            ),
            "pose_bbox_xyxy_img": _Array(
                path="keypoints_runs/source/pose_bbox_xyxy_img", values=bbox_img
            ),
        },
    )
    plan = keypoint_successor._plan_keypoint_auxiliaries(
        source_run=source,
        source_manifest=source_manifest,
        sealed_crop=binding,
    )
    assert set(plan.values) == {
        "source_crop_xywh",
        "keypoints_norm",
        "pose_bbox_xyxy_norm",
    }
    assert plan.as_record()["logical_payload_scope"]["auxiliary_arrays_excluded"]
    target_root = zarr.open_group(str(tmp_path / "target.zarr"), mode="w")
    target = target_root.create_group("keypoints_runs").create_group("successor")
    payload = np.asarray([[101, 102]], dtype="<f4")
    target.create_array("keypoints_roi", data=payload)
    receipt = keypoint_successor._materialize_keypoint_auxiliaries(
        target, plan=plan, source_manifest=source_manifest
    )
    assert receipt["policy"] == plan.as_record()["policy"]
    assert set(receipt["arrays"]) == set(plan.values)
    for name, values in plan.values.items():
        assert np.array_equal(target[name][...], values, equal_nan=True)
        assert target[name].attrs["coordinate_successor_auxiliary"] is True
    assert np.array_equal(target["keypoints_roi"][...], payload)


def test_actual_keypoint_prepare_and_publish_accept_explicit_padded_crop_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the unmocked coordinate publisher with a padded requested window."""

    root, run, _crop, _roi_images = _keypoint_publication_fixture(monkeypatch)
    canonical_source = (
        keypoint_coordinate_publication.load_persisted_keypoint_crop_source(
            root, "crop_runs/c1"
        )
    )
    canonical_geometry = canonical_source.crop_geometry
    canonical_frames = canonical_geometry.source_geometry.frame_evidence
    source = sealed_crop.BoundSealedGeometryCropSource(
        crop_geometry=SimpleNamespace(
            source_geometry=SimpleNamespace(
                frame_evidence=SimpleNamespace(
                    source_camera_frame=canonical_frames.source_camera_frame,
                    bbox_source_camera_frame=canonical_frames.bbox_source_camera_frame,
                    acquisition_frame=canonical_frames.acquisition_frame,
                    normalized_frame=SimpleNamespace(
                        record_ref="/crop_runs/c1@historical_manifest",
                        record_sha256="6" * 64,
                    ),
                    normalized_to_source_camera=SimpleNamespace(transform_records=()),
                )
            ),
            row_identity=canonical_geometry.row_identity,
            selection_derivation=canonical_geometry.selection_derivation,
        ),
        roi_geometry=canonical_source.roi_geometry,
        roi_frame=canonical_source.roi_frame,
        bbox_roi_frame=canonical_source.bbox_roi_frame,
        crop_path=canonical_source.crop_path,
        crop_profile="sealed_geometry_only_v2",
        placement_ownership_attr=CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
        placement_pixel_center_ownership_attr=(
            CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR
        ),
        placement_pixel_edge_ownership_attr=(
            CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR
        ),
        _root=canonical_source._root,
        _rowset_node=canonical_source._rowset_node,
        _placement_node=canonical_source._placement_node,
        _roi_images_node=None,
        _verification_seal=sealed_crop._BOUND_SEALED_GEOMETRY_CROP_SOURCE_SEAL,
    )

    source_rows = np.asarray(run["source_crop_row_ids"][:])
    placements = np.asarray(run["source_crop_xywh"][:], dtype="<f4").copy()
    placements[0, 0] = -5.0
    run["source_crop_xywh"].data = placements
    run["source_crop_xywh"].dtype = placements.dtype
    crop_placements = np.asarray(source._placement_node[:], dtype="<f4").copy()
    crop_placements[source_rows] = placements
    source._placement_node.data = crop_placements
    source._placement_node.dtype = crop_placements.dtype

    offsets = placements[:, :2]
    keypoints_roi = np.asarray(run["keypoints_roi"][:])
    keypoints_img = keypoints_roi + offsets[:, None, :]
    keypoints_img[~np.isfinite(keypoints_roi)] = np.nan
    run["keypoints_img"].data = keypoints_img
    run["keypoints_norm"].data = keypoints_img / np.asarray([100.0, 80.0])

    bbox_roi = np.asarray(run["pose_bbox_xyxy_roi"][:])
    bbox_img = bbox_roi + np.column_stack((offsets, offsets))
    run["pose_bbox_xyxy_img"].data = bbox_img
    run["pose_bbox_xyxy_norm"].data = np.asarray(
        bbox_img / np.asarray([100.0, 80.0, 100.0, 80.0]),
        dtype=bbox_roi.dtype,
    )

    padded_lineage = sealed_crop._padded_lineage_summary(
        placements,
        source_width=100,
        source_height=80,
        roi_width=40,
        roi_height=40,
        crop_policy_payload_digest="a" * 64,
        origin_authority_digest="b" * 64,
        provider_record_sha256="c" * 64,
    )
    binding = sealed_crop.SealedGeometryCropSuccessorBinding(
        source=source,
        crop_path="crop_runs/c1",
        manifest={},
        manifest_digest="1" * 64,
        manifest_payload_digest="2" * 64,
        logical_content_digest="3" * 64,
        row_signatures_digest="4" * 64,
        coordinate_catalog_digest="5" * 64,
        n_frames=2,
        n_instances=2,
        source_width=100,
        source_height=80,
        source_run_path="keypoints_runs/source",
        padded_lineage=padded_lineage,
        successor_evidence_record={
            "schema_id": sealed_crop.SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_SCHEMA_ID,
            "schema_version": sealed_crop.SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_SCHEMA_VERSION,
            "adapter_kind": sealed_crop.SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_KIND,
            "crop_path": "crop_runs/c1",
        },
    )
    sealed_crop.stamp_persisted_padded_placement_provenance(run, binding)
    publication_binding = (
        sealed_crop.bind_sealed_geometry_bbox_normalization_to_successor(
            binding,
            root=root,
            successor_run=run,
            successor_run_path="keypoints_runs/k1",
        )
    )

    transform = ModelInputTransform(
        name="identity",
        native_height=40,
        native_width=40,
        model_height=40,
        model_width=40,
    )
    monkeypatch.setattr(
        sealed_crop,
        "require_bound_sealed_geometry_crop_source",
        lambda value, **_kwargs: value,
    )
    keypoint_coordinate_publication.prepare_keypoint_coordinate_context(
        root,
        "keypoints_runs/k1",
        crop_path="crop_runs/c1",
        model_input_transform=transform,
        preprocessing_input_mode="numpy-list",
        model_artifact=_keypoint_artifact(),
        _resolved_crop_source=publication_binding.source,
    )
    surfaces = keypoint_coordinate_publication.publish_keypoint_coordinate_surfaces(
        root,
        "keypoints_runs/k1",
        _resolved_crop_source=publication_binding.source,
    )

    ownership = run["source_crop_xywh"].attrs[CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR]
    assert ownership["schema_id"] == CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_ID
    assert ownership["window_geometry"]["padded_row_count"] == 1
    legacy_ownership = deepcopy(ownership)
    legacy_ownership["producer"] = CROP_PLACEMENT_PADDED_LEGACY_PRODUCER
    assert parse_crop_placement_ownership(legacy_ownership).producer == (
        CROP_PLACEMENT_PADDED_LEGACY_PRODUCER
    )
    assert surfaces.source_crop_xywh.descriptor.source_camera_overlay.status == "direct"
    bbox_evidence = run.attrs[sealed_crop.SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR]
    assert bbox_evidence["normalized_frame"]["record_ref"].startswith(
        "/keypoints_runs/k1/coordinate_frames/"
    )
    assert (
        surfaces.pose_bbox_xyxy_norm.reference_frame_authority.record_ref
        == bbox_evidence["normalized_frame"]["record_ref"]
    )
    run.attrs["status"] = RUN_STATUS_COMPLETE
    run.attrs[RUN_COMPLETION_STATUS_ATTR] = RUN_STATUS_COMPLETE
    reloaded_binding = (
        sealed_crop.load_sealed_geometry_bbox_normalization_from_successor(
            binding,
            root=root,
            successor_run=run,
            successor_run_path="keypoints_runs/k1",
        )
    )
    reloaded_evidence = (
        reloaded_binding.source.crop_geometry.source_geometry.frame_evidence
    )
    assert (
        reloaded_evidence.normalized_frame.record_ref
        == bbox_evidence["normalized_frame"]["record_ref"]
    )
    assert [
        item.record_ref
        for item in reloaded_evidence.normalized_to_source_camera.transform_records
    ] == [
        item["record_ref"]
        for item in bbox_evidence["normalized_to_source_camera"]
    ]
    assert not hasattr(sealed_crop, "historical_geometry_only_crop_loader")


def test_actual_subject_mask_prepare_and_publish_accept_explicit_padded_crop_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all unmocked subject-mask placement conventions with padding."""

    root, _parent, run, source = _subject_fixture_with_source(monkeypatch, fresh=True)
    source = copy(source)
    object.__setattr__(
        source,
        "placement_ownership_attr",
        CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
    )
    object.__setattr__(
        source,
        "placement_pixel_center_ownership_attr",
        CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
    )
    object.__setattr__(
        source,
        "placement_pixel_edge_ownership_attr",
        CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
    )
    source_rows = np.asarray(run["source_crop_row_ids"][:])
    placements = np.asarray(run["source_crop_xywh"][:], dtype="<f4").copy()
    placements[0, 1] = -7.0
    run["source_crop_xywh"].data = placements
    run["source_crop_xywh"].dtype = placements.dtype
    crop_placements = np.asarray(source._placement_node[:], dtype="<f4").copy()
    crop_placements[source_rows] = placements
    source._placement_node.data = crop_placements
    source._placement_node.dtype = crop_placements.dtype

    padded_lineage = sealed_crop._padded_lineage_summary(
        placements,
        source_width=100,
        source_height=80,
        roi_width=40,
        roi_height=40,
        crop_policy_payload_digest="d" * 64,
        origin_authority_digest="e" * 64,
        provider_record_sha256="f" * 64,
    )
    binding = SimpleNamespace(
        source=source,
        crop_path="crop_runs/c1",
        padded_lineage=padded_lineage,
    )
    sealed_crop.stamp_persisted_padded_placement_provenance(run, binding)

    owner = _subject_mask_owner(run)
    monkeypatch.setattr(
        keypoint_coordinate_publication,
        "require_bound_keypoint_crop_source",
        lambda value: value,
    )
    subject_mask_coordinate_publication.prepare_subject_mask_coordinate_context(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=owner,
        crop_path="crop_runs/c1",
        mask_labels=SUBJECT_MASK_LABELS,
        model_input_transform=SUBJECT_MASK_MODEL_TRANSFORM,
        model_artifact=SUBJECT_MASK_MODEL_ARTIFACT,
        mask_probability_threshold=0.5,
        _resolved_crop_source=source,
    )
    surfaces = subject_mask_coordinate_publication.publish_subject_mask_coordinate_surfaces(
        root,
        "subject_mask_runs/s1",
        expected_publication_owner=owner,
        _resolved_crop_source=source,
    )

    placement_attrs = run["source_crop_xywh"].attrs
    for attr_name in (
        CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
        CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
        CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
    ):
        assert placement_attrs[attr_name]["schema_id"] == (
            CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_ID
        )
        assert placement_attrs[attr_name]["window_geometry"]["padded_row_count"] == 1
    assert surfaces.context.run_path == "subject_mask_runs/s1"


@pytest.mark.parametrize("prepare_mismatch", [False, True])
def test_keypoint_successor_apply_reaches_preparation_with_padded_auxiliaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepare_mismatch: bool,
) -> None:
    """Exercise the publisher boundary with a real temporary Zarr successor.

    The harness keeps scientific readers and the source archive minimal, but it
    calls the production publisher, copy/materialization path, authority
    stamping, and failure marking.  The preparation spy is intentionally the
    regression point for both historical failures: it requires the missing
    ``source_crop_xywh`` auxiliary and the successor-owned padded provenance
    before the preparation call can proceed.
    """

    archive = tmp_path / "analysis.zarr"
    root = zarr.open_group(str(archive), mode="w")
    family = root.create_group("keypoints_runs")
    source = family.create_group("source")
    family.attrs["latest"] = "source"
    source_arrays = {
        "source_crop_row_ids": np.asarray([0, 1], dtype="<i8"),
        "instance_key": np.asarray([10, 11], dtype="<u8"),
        "source_acquisition_frame_index": np.asarray([0, 1], dtype="<i8"),
        "frame_indices": np.asarray([0, 1], dtype="<i8"),
        "frame_row_offsets": np.asarray([0, 1, 2], dtype="<i8"),
        "source_crop_row_signature": np.zeros((2, 32), dtype="uint8"),
        "keypoint_row_signature": np.ones((2, 32), dtype="uint8"),
        "keypoints_roi": np.asarray([[[1, 2]], [[3, 4]]], dtype="<f4"),
        "keypoints_img": np.asarray([[[1, 2]], [[3, 4]]], dtype="<f4"),
        "keypoint_confidences": np.asarray([[0.8], [0.9]], dtype="<f4"),
        "keypoint_valid": np.asarray([[True], [True]], dtype="bool"),
        "pose_confidence": np.asarray([0.8, 0.9], dtype="<f4"),
        "pose_bbox_xyxy_roi": np.asarray([[1, 2, 10, 20], [3, 4, 12, 22]], dtype="<f4"),
        "pose_bbox_xyxy_img": np.asarray([[1, 2, 10, 20], [3, 4, 12, 22]], dtype="<f4"),
        "pose_success": np.asarray([True, True], dtype="bool"),
    }
    for name, values in source_arrays.items():
        source.create_array(name, data=values)
    transform = ModelInputTransform(
        name="identity",
        native_height=384,
        native_width=384,
        model_height=384,
        model_width=384,
    )
    model_artifact = _keypoint_artifact(keypoint_labels=("swim_bladder",))
    source_manifest = {
        "schema_id": "palette.keypoint.run_manifest",
        "schema_version": 1,
        "payload_digest": "1" * 64,
        "payload": {
            "run_id": "source",
            "logical_content": {"digest": "2" * 64},
            "logical_schema": {
                "dimensions": {
                    "n_frames": 2,
                    "n_instances": 2,
                    "n_keypoints": 1,
                    "source_width": 4512,
                    "source_height": 4512,
                }
            },
            "storage_plan": {"storage_profile": PUBLISHED_HTTP_V1.as_manifest()},
            "preprocessing": {},
            "source_crop_snapshot": {"run_path": "crop_runs/crop"},
            "pose_model_schema_binding": model_artifact["pose_schema_binding"],
        },
    }
    source.attrs.update(
        {
            "run_manifest": source_manifest,
            "status": "complete",
            "stage_selector_eligible": False,
            "production_candidate": True,
        }
    )
    source_metadata_digest = metadata_tree_sha256(archive / "keypoints_runs/source")
    selectors = keypoint_successor._selector_snapshot(root)
    values = {
        "source_crop_xywh": np.asarray(
            [[-1, 2, 384, 384], [3, 4, 384, 384]], dtype="<f4"
        ),
        "keypoints_norm": np.asarray([[[0.1, 0.2]], [[0.3, 0.4]]], dtype="<f4"),
        "pose_bbox_xyxy_norm": np.asarray(
            [[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6]], dtype="<f4"
        ),
    }
    storage = {
        name: keypoint_successor._auxiliary_array_contract_and_plan(
            name, array, profile=PUBLISHED_HTTP_V1
        )[1].as_dict()
        for name, array in values.items()
    }
    auxiliary_plan = keypoint_successor._KeypointAuxiliaryMaterializationPlan(
        values=values,
        storage=storage,
        source_existing={},
        source_crop_row_ids_sha256=keypoint_successor.sha256_array(
            source_arrays["source_crop_row_ids"]
        ),
    )
    padded_lineage = sealed_crop._padded_lineage_summary(
        values["source_crop_xywh"],
        source_width=4512,
        source_height=4512,
        roi_width=384,
        roi_height=384,
        crop_policy_payload_digest="a" * 64,
        origin_authority_digest="b" * 64,
        provider_record_sha256="c" * 64,
    )
    binding = SimpleNamespace(
        source=object(),
        padded_lineage=padded_lineage,
        as_record=lambda: {"adapter_kind": "test_historical_adapter"},
    )
    preprocessing = SimpleNamespace(
        document={"model_input_transform": transform.to_attrs()}
    )
    surfaces = SimpleNamespace(
        context=SimpleNamespace(
            context_record=SimpleNamespace(
                record_ref="/keypoints_runs/successor@context", record_sha256="3" * 64
            ),
            row_identity=SimpleNamespace(
                record_ref="/keypoints_runs/successor@rows", record_sha256="4" * 64
            ),
            temporal_authority=SimpleNamespace(
                record_ref="/keypoints_runs/successor@temporal", record_sha256="5" * 64
            ),
        ),
        derivation=SimpleNamespace(
            record_ref="/keypoints_runs/successor@derivation", record_sha256="6" * 64
        ),
    )

    def fake_prepare(target_root, target_path, **kwargs):
        assert kwargs["_resolved_crop_source"] is binding.source
        target = target_root[target_path]
        assert "source_crop_xywh" in target
        assert target.attrs["keypoint_labels"] == ["swim_bladder"]
        assert CROP_PLACEMENT_PADDED_PROVENANCE_ATTR in target["source_crop_xywh"].attrs
        if prepare_mismatch:
            raise ValueError("synthetic padded placement mismatch")
        ownership = {
            "schema_id": CROP_PLACEMENT_PADDED_OWNERSHIP_SCHEMA_ID,
            "schema_version": 1,
            "producer": CROP_PLACEMENT_PADDED_PRODUCER,
            "layout": "xywh",
            "window_policy": "requested_window_zero_padded_v1",
            "crop_placement": {"record_ref": "/target/source_crop_xywh"},
            "row_identity": {"record_ref": "/target/rows"},
            "source_camera_frame": {"record_ref": "/target/camera"},
            "camera_id": "cam",
            "canonicalization": "canonical_json_sort_keys_v1",
            "window_geometry": {
                **{
                    key: value
                    for key, value in padded_lineage.items()
                    if key
                    not in {"schema_id", "schema_version", "crop_policy_provenance"}
                },
                "schema_id": "palette.coordinate_successor.padded_crop_window_geometry",
                "schema_version": 1,
                "requested_extent": {"width": 384, "height": 384, "units": "px"},
                "crop_policy_provenance": padded_lineage["crop_policy_provenance"],
            },
        }
        placement_attrs = target["source_crop_xywh"].attrs
        placement_attrs[CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR] = ownership
        placement_attrs[f"{CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR}_sha256"] = (
            canonical_json_sha256(ownership)
        )
        bound_records = {}
        for name in ("context", "rows", "temporal", "derivation"):
            bound_records[name] = stamp_and_bind_persisted_coordinate_record(
                target,
                {"schema_id": f"test.{name}", "schema_version": 1},
                attr_name=name,
            )
        surfaces.context.context_record = bound_records["context"]
        surfaces.context.row_identity = bound_records["rows"]
        surfaces.context.temporal_authority = bound_records["temporal"]
        surfaces.derivation = bound_records["derivation"]
        assert "context" in target.attrs

    monkeypatch.setattr(
        keypoint_successor,
        "inspect_keypoint_coordinate_successor_source",
        lambda **_kwargs: {
            "analysis_zarr": str(archive),
            "source_run_id": "source",
            "successor_run_id": "successor",
            "source_manifest_digest": canonical_json_sha256(source_manifest),
            "source_metadata_tree_sha256": source_metadata_digest,
            "source_logical_content_digest": "2" * 64,
            "selectors_before": selectors,
            "historical_crop_adapter": binding.as_record(),
            "auxiliary_materialization": auxiliary_plan.as_record(),
            "keypoint_semantic_attrs": keypoint_successor._keypoint_semantic_attrs(
                source_manifest["payload"], model_artifact=model_artifact
            ),
        },
    )
    monkeypatch.setattr(
        keypoint_successor,
        "_source_bundle_authority",
        lambda *_args, **_kwargs: {"schema_id": "test"},
    )
    monkeypatch.setattr(
        keypoint_successor,
        "keypoint_preprocessing_from_manifest",
        lambda _value: preprocessing,
    )
    monkeypatch.setattr(
        keypoint_successor,
        "_resolve_preprocessing_runtime",
        lambda _value: (transform, "tensor"),
    )
    monkeypatch.setattr(
        keypoint_successor,
        "_model_artifact",
        lambda *_args, **_kwargs: model_artifact,
    )
    monkeypatch.setattr(
        keypoint_successor,
        "bind_sealed_geometry_crop_successor_source",
        lambda **_kwargs: binding,
    )

    def fake_bind_bbox_normalization(bound, *, root, successor_run, successor_run_path):
        del root
        target = successor_run
        record = {
            "schema_id": sealed_crop.SEALED_GEOMETRY_BBOX_NORMALIZATION_SCHEMA_ID,
            "schema_version": sealed_crop.SEALED_GEOMETRY_BBOX_NORMALIZATION_SCHEMA_VERSION,
            "successor_run_path": successor_run_path,
        }
        target.attrs[sealed_crop.SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR] = record
        target.attrs[f"{sealed_crop.SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR}_sha256"] = (
            canonical_json_sha256(record)
        )
        return bound

    monkeypatch.setattr(
        keypoint_successor,
        "bind_sealed_geometry_bbox_normalization_to_successor",
        fake_bind_bbox_normalization,
    )
    monkeypatch.setattr(
        keypoint_successor,
        "_plan_keypoint_auxiliaries",
        lambda **_kwargs: auxiliary_plan,
    )
    monkeypatch.setattr(
        keypoint_successor, "prepare_keypoint_coordinate_context", fake_prepare
    )

    def fake_publish(target_root, target_path, **kwargs):
        assert kwargs["_resolved_crop_source"] is binding.source
        fresh_target = target_root[target_path]
        fresh_target.attrs["coordinate_contract"] = "canonical_v2"
        surfaces.context._run_group = fresh_target
        return surfaces

    monkeypatch.setattr(
        keypoint_successor,
        "publish_keypoint_coordinate_surfaces",
        fake_publish,
    )
    monkeypatch.setattr(
        keypoint_successor,
        "load_persisted_ineligible_keypoint_coordinate_surfaces",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        keypoint_successor,
        "require_bound_ineligible_keypoint_coordinate_surfaces",
        lambda value: value,
    )
    monkeypatch.setattr(
        keypoint_successor,
        "keypoint_metadata_declaration_maps",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        keypoint_successor, "plan_keypoint_storage", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        keypoint_successor,
        "build_keypoint_coordinate_successor_manifest",
        lambda manifest, **_kwargs: deepcopy(manifest),
    )
    monkeypatch.setattr(
        keypoint_successor,
        "validate_keypoint_publication",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        keypoint_successor,
        "_source_crop_manifest_and_arrays",
        lambda *_args: (None, {}),
    )
    monkeypatch.setattr(
        keypoint_successor,
        "consolidate_metadata_capture_expected_warnings",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        keypoint_successor,
        "load_coordinate_successor_authority",
        lambda run, **_kwargs: run.attrs["coordinate_successor_authority"],
    )

    source_before = metadata_tree_sha256(archive / "keypoints_runs/source")
    if prepare_mismatch:
        with pytest.raises(ValueError, match="synthetic padded placement mismatch"):
            keypoint_successor.publish_keypoint_coordinate_successor(
                analysis_zarr=archive,
                source_run_id="source",
                successor_run_id="successor",
                keypoint_model_path=tmp_path / "unused.pt",
            )
        failed_root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
        assert failed_root["keypoints_runs/successor"].attrs["status"] == "failed"
    else:
        result = keypoint_successor.publish_keypoint_coordinate_successor(
            analysis_zarr=archive,
            source_run_id="source",
            successor_run_id="successor",
            keypoint_model_path=tmp_path / "unused.pt",
        )
        assert result["status"] == "complete"
        assert set(result["auxiliary_materialization"]["arrays"]) == set(values)
        published = zarr.open_group(str(archive), mode="r", use_consolidated=False)
        target = published["keypoints_runs/successor"]
        assert (
            target.attrs["coordinate_successor_policy"]
            == keypoint_successor.KEYPOINT_COORDINATE_SUCCESSOR_PUBLICATION_POLICY
        )
        assert CROP_PLACEMENT_PADDED_PROVENANCE_ATTR in target["source_crop_xywh"].attrs
        assert target.attrs["coordinate_contract"] == "canonical_v2"
    assert (
        keypoint_successor._selector_snapshot(
            zarr.open_group(str(archive), mode="r", use_consolidated=False)
        )
        == selectors
    )
    assert metadata_tree_sha256(archive / "keypoints_runs/source") == source_before


def test_padded_ownership_parser_requires_explicit_window_geometry() -> None:
    record = {
        "schema_id": "palette.coordinate_successor.padded_crop_placement_ownership",
        "schema_version": 1,
        "producer": CROP_PLACEMENT_PADDED_PRODUCER,
        "layout": "xywh",
        "window_policy": "requested_window_zero_padded_v1",
        "crop_placement": {},
        "row_identity": {},
        "source_camera_frame": {},
        "camera_id": "cam",
        "canonicalization": "canonical_json_sort_keys_v1",
    }
    with pytest.raises(Exception, match="window_geometry"):
        parse_crop_placement_ownership(record)
