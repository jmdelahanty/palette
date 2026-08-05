from __future__ import annotations

from copy import deepcopy
import hashlib
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest
import zarr

import fisheye.shared.zarr.subject_mask_core_publication as core_publication
from fisheye.shared.subject_mask_attempt import (
    build_subject_mask_attempt,
    build_subject_mask_scientific_identity,
)
from fisheye.shared.subject_mask_worker_receipt import (
    RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS,
    build_recording_subject_mask_source_receipt,
    build_subject_mask_worker_semantic_receipt,
)
from fisheye.shared.zarr.crop_manifest import build_coordinate_crop_run_manifest
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR,
    SubjectMaskCoreValidationMode,
    build_subject_mask_core_coordinate_dependencies,
    publish_selector_ineligible_subject_mask_core_snapshot,
    validate_subject_mask_core_run_manifest,
)
from tests.unit.fisheye.test_crop_manifest import (
    _arrays as _crop_manifest_arrays,
    _dimensions as _crop_dimensions,
    _metadata as _crop_metadata,
    _pixel as _crop_pixel,
    _policy as _crop_policy,
    _source as _crop_source,
)
from fisheye.shared.zarr.subject_mask_schema import (
    RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
    REFINED_SUBJECT_MASK_CORE_SCHEMA_V1,
    SubjectMaskComponentRegistry,
    SubjectMaskDimensions,
    derive_subject_mask_frame_row_offsets,
    derive_subject_mask_metrics,
)
from fisheye.shared.zarr.subject_mask_validation_receipt import (
    SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID,
    SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION,
    SubjectMaskArrayUnitAccumulator,
    build_reference_subject_mask_validation_receipt,
    build_subject_mask_source_validation_receipt,
    subject_mask_array_unit_document,
    subject_mask_semantic_units_from_array_document,
    validate_subject_mask_source_validation_receipt,
)


def test_incremental_row_hashes_ignore_execution_batch_boundaries() -> None:
    values = np.arange(7 * 2, dtype=np.uint16).reshape(7, 2)
    accumulator = SubjectMaskArrayUnitAccumulator(
        shape=values.shape,
        dtype=values.dtype,
        unit_rows=3,
    )
    accumulator.append(0, values[:2])
    accumulator.append(2, values[2:6])
    accumulator.append(6, values[6:])

    document = accumulator.as_document()
    expected = subject_mask_array_unit_document(
        {"values": values}, ("values",), unit_rows=3
    )["values"]

    assert document == expected
    semantic = subject_mask_semantic_units_from_array_document(
        {"values": document}, n_rois=7, paths=("values",)
    )
    assert [(unit["start_row"], unit["stop_row"]) for unit in semantic] == [
        (0, 3),
        (3, 6),
        (6, 7),
    ]


def test_incremental_row_hashes_reject_gaps_and_dtype_changes() -> None:
    accumulator = SubjectMaskArrayUnitAccumulator(
        shape=(3, 2), dtype=np.uint16, unit_rows=2
    )
    with pytest.raises(ValueError, match="not contiguous"):
        accumulator.append(1, np.zeros((1, 2), dtype=np.uint16))
    with pytest.raises(ValueError, match="dtype differs"):
        accumulator.append(0, np.zeros((1, 2), dtype=np.uint8))


def _components() -> SubjectMaskComponentRegistry:
    return SubjectMaskComponentRegistry(
        ("subject_body", "eye_left", "eye_right", "swim_bladder")
    )


def _fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    masks = np.zeros((4, 4, 8, 8), dtype=np.uint8)
    masks[:, 0, 1:7, 1:7] = 1
    masks[:, 1, 2, 2] = 1
    masks[:, 2, 2, 5] = 1
    masks[:, 3, 5, 3] = 1
    metrics = derive_subject_mask_metrics(masks)
    frames = np.asarray([0, 0, 2, 3], dtype=np.int64)
    common = {
        "source_crop_row_ids": np.arange(4, dtype=np.int64),
        "instance_key": np.asarray([101, 102, 201, 301], dtype=np.uint64),
        "source_acquisition_frame_index": frames,
        "frame_row_offsets": derive_subject_mask_frame_row_offsets(frames, n_frames=4),
        "source_crop_xywh": np.asarray(
            [[16, 12, 8, 8], [66, 12, 8, 8], [46, 52, 8, 8], [21, 56, 8, 8]],
            dtype=np.float32,
        ),
        "available_channels": np.ones((4,), dtype=bool),
        **{f"metrics/{name}": values for name, values in metrics.items()},
    }
    probabilities = masks * np.uint8(255)
    raw = {
        **common,
        "mask_probs_roi": probabilities,
        "metrics/prob_max": np.max(
            probabilities.astype(np.float32) / np.float32(255.0),
            axis=(2, 3),
        ).astype(np.float32),
    }
    refined = {**common, "masks_roi": masks}
    crop = {
        "instance_key": common["instance_key"],
        "source_acquisition_frame_index": frames,
        "source_crop_xywh": common["source_crop_xywh"],
    }
    return raw, {**refined, "_crop": crop}  # type: ignore[dict-item]


def _dimensions() -> SubjectMaskDimensions:
    return SubjectMaskDimensions(
        n_frames=4,
        n_rois=4,
        n_channels=4,
        roi_height=8,
        roi_width=8,
    )


def _coordinate_crop_manifest() -> dict[str, object]:
    plan, direct, consolidated = _crop_metadata()
    return build_coordinate_crop_run_manifest(
        run_id="crop_coordinate_shadow",
        dimensions=_crop_dimensions(),
        policy=_crop_policy(),
        storage_plan=plan,
        arrays=_crop_manifest_arrays(),
        source=_crop_source(),
        pixel_authority=_crop_pixel(),
        direct_metadata_declarations=direct,
        consolidated_metadata_declarations=consolidated,
        selector_eligible=False,
    )


def _collection_partition_contract() -> dict[str, object]:
    payload = {
        "role": "complete_collection_partition",
        "coverage_semantics": (
            "exact_complete_crop_rows_for_acquisition_frame_window_v1"
        ),
        "work_package_id": "d" * 64,
        "collection": {
            "source_collection_id": "pytest_collection",
            "source_collection_path": "/pytest/collection.json",
            "source_clip_id": "whole_recording",
            "source_clip_index": 0,
            "source_work_unit_id": "pytest_collection:whole_recording",
            "source_shard_id": "whole_recording",
        },
        "frame_window": {
            "schema_id": "palette.acquisition_video_frame_window",
            "schema_version": 1,
            "recording_identity": "crop_manifest_test",
            "camera_identity": "cam2010095",
            "clip_id": "whole_recording",
            "actual_start_frame": 0,
            "end_frame_exclusive": 4,
            "frame_count": 4,
            "clip_index_document_sha256": "e" * 64,
            "clip_video_sha256": "f" * 64,
        },
        "crop_rows": {
            "start": 0,
            "stop": 4,
            "count": 4,
            "source_crop_total_rows": 4,
        },
        "validation": {
            "work_package_opened_and_content_verified": True,
            "row_interval_contiguous": True,
            "frame_offset_coverage_exact": True,
            "acquisition_frames_within_window": True,
        },
    }
    return {
        "schema_id": "palette.subject_mask.complete_collection_partition",
        "schema_version": 1,
        "payload": payload,
        "payload_digest": canonical_json_sha256(payload),
    }


def _recording_documents(
    *,
    kind: str,
    arrays: dict[str, np.ndarray],
    source_producer_evidence: dict[str, object] | None = None,
    source_producer_run_path: str | None = None,
) -> tuple[dict[str, object], dict[str, object], str]:
    raw = kind == "raw_probability_uint8"
    stage = "raw_subject_mask" if raw else "refined_subject_mask"
    schema = (
        RAW_SUBJECT_MASK_UINT8_SCHEMA_V1 if raw else REFINED_SUBJECT_MASK_CORE_SCHEMA_V1
    )
    worker_paths = (
        RAW_SUBJECT_MASK_WORKER_OUTPUT_PATHS
        if raw
        else REFINED_SUBJECT_MASK_WORKER_OUTPUT_PATHS
    )
    worker_path = (
        "subject_mask_shard_runs/worker_001"
        if raw
        else "refined_subject_masks_runs/worker_001"
    )
    source_path = (
        "subject_mask_shard_collections/recording_001"
        if raw
        else "refined_subject_mask_shard_collections/recording_001"
    )

    def array_reference(value: np.ndarray) -> dict[str, object]:
        return {
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(
                np.ascontiguousarray(value).view(np.uint8)
            ).hexdigest(),
        }

    crop_manifest = _coordinate_crop_manifest()
    if raw:
        science = build_subject_mask_scientific_identity(
            stage_kind=stage,
            model={
                "artifact_role": "subject_mask_checkpoint",
                "artifact_sha256": "a" * 64,
                "artifact_size_bytes": 1024,
                "registry_set_id": "pytest_models",
                "registry_run_id": "pytest_raw",
                "label_schema_id": "pytest_subject_masks",
            },
            crop={
                "run_id": "crop_coordinate_shadow",
                "run_group_path": "crop_runs/crop_coordinate_shadow",
                "run_manifest": {
                    "schema_id": crop_manifest["schema_id"],
                    "schema_version": crop_manifest["schema_version"],
                    "payload_digest": crop_manifest["payload_digest"],
                },
                "storage_mode": "geometry_only",
                "roi_shape_hw": [8, 8],
                "roi_coordinates_full": array_reference(
                    arrays["source_crop_xywh"][:, :2].astype(np.int32)
                ),
                "source_collection_id": "pytest_collection",
                "source_clip_id": "whole_recording",
                "source_clip_index": 0,
                "source_work_unit_id": "pytest_collection:whole_recording",
                "source_shard_id": "whole_recording",
                "collection_partition_contract": _collection_partition_contract(),
            },
            pixels={
                "profile": "pytest_pixels",
                "decoded_shape": [4, 8, 8],
                "decoded_dtype": "uint8",
                "decoded_order": "C",
                "decoded_pixels_sha256": "b" * 64,
                "declared_pixels_sha256": "b" * 64,
                "cache_key": "pytest_cache",
                "pixel_materialization_id": "pytest_pixels",
                "pixel_contract": {"schema": "palette_roi_pixel_contract_v1"},
                "work_package_role": "complete_collection_partition",
            },
            row_identity={
                "row_count": 4,
                "arrays": {
                    name: array_reference(arrays[name])
                    for name in (
                        "source_crop_row_ids",
                        "instance_key",
                        "source_acquisition_frame_index",
                    )
                },
            },
            inference_contract={
                "segmenter": "unet",
                "label_schema_id": "pytest_subject_masks",
                "mask_labels": list(_components().labels),
                "model_input_transform": {
                    "name": "identity",
                    "native_shape_hw": [8, 8],
                    "model_shape_hw": [8, 8],
                    "pad_top": 0,
                    "pad_bottom": 0,
                    "pad_left": 0,
                    "pad_right": 0,
                    "coordinate_mapping": (
                        "native_xy = model_xy - [pad_left, pad_top]"
                    ),
                },
                "probability_semantics": "sigmoid_multilabel_logits",
                "probability_dtype": "uint8",
                "probability_encoding": "linear_uint8_0_255",
                "mask_probability_threshold": 0.5,
                "overlap_policy": "independent_sigmoid",
            },
        )
    else:
        assert source_producer_evidence is not None
        raw_worker = source_producer_evidence["workers"][0]
        raw_receipt = raw_worker["worker_receipt"]
        input_binding = {
            "run_path": raw_worker["run_path"],
            "run_manifest": None,
            "scientific_identity_digest": raw_worker["scientific_identity_digest"],
            "worker_semantic_receipt_binding": {
                "schema_id": raw_receipt["schema_id"],
                "schema_version": raw_receipt["schema_version"],
                "payload_digest": raw_receipt["payload_digest"],
                "relative_path": "pytest/raw_worker_semantic_receipt.json",
                "document_sha256": canonical_json_sha256(raw_receipt),
                "storage": "strict_json_sidecar_v1",
            },
        }
        method = "smart_finalize_subject_masks_v1"
        science = build_subject_mask_scientific_identity(
            stage_kind=stage,
            model={
                "role": "deterministic_refinement_policy",
                "method": method,
                "source_input_binding": input_binding,
            },
            crop={
                "run_id": "crop_coordinate_shadow",
                "source_crop_snapshot": {},
                "roi_shape_hw": [8, 8],
            },
            pixels={
                "semantic_input": "raw_subject_mask_surface",
                "surface_kind": "probability",
                "surface_path": "mask_probs_roi",
                "probability_encoding": "linear_uint8_0_255",
                "source_input_binding": input_binding,
            },
            row_identity={
                "row_count": 4,
                "arrays": {
                    name: array_reference(arrays[name])
                    for name in (
                        "source_crop_row_ids",
                        "instance_key",
                        "source_acquisition_frame_index",
                        "source_crop_xywh",
                        "available_channels",
                    )
                },
            },
            inference_contract={
                "method": method,
                "finalization_semantics": ("smart_probability_to_refined_candidate"),
                "output_component_order": list(_components().labels),
                "component_sources_and_policies": {
                    label: {"source": "pytest"} for label in _components().labels
                },
                "eye_assignment_contract": None,
                "authoritative_output": "dense_uint8_masks_roi",
                "derived_cache_policy": (
                    "bitpacked_rle_metrics_contours_non_authoritative"
                ),
            },
        )
    attempt = build_subject_mask_attempt(
        scientific_identity=science,
        run_path=worker_path,
        attempt_id=str(uuid5(NAMESPACE_URL, f"pytest:{kind}")),
    )
    local = {path: arrays[path] for path in worker_paths}
    receipt = build_subject_mask_worker_semantic_receipt(
        stage_kind=stage,
        run_path=worker_path,
        scientific_identity=science,
        attempt=attempt,
        scope={"recording": "crop_manifest_test"},
        row_count=4,
        array_document=subject_mask_array_unit_document(
            local,
            worker_paths,
            unit_rows=2,
        ),
        required_paths=worker_paths,
        roi_aligned_paths=tuple(
            path for path in worker_paths if path != "available_channels"
        ),
    )
    manifest, source_receipt = build_recording_subject_mask_source_receipt(
        kind=kind,
        stage_kind=stage,
        source_run_path=source_path,
        schema=schema,
        arrays=arrays,
        dimensions=_dimensions(),
        components=_components(),
        threshold=0.5 if raw else None,
        workers=(
            {
                "global_start_row": 0,
                "scientific_identity": science,
                "attempt": attempt,
                "receipt": receipt,
            },
        ),
        identity_unit_rows=2,
        expected_work_units=(
            {
                "work_unit_id": "pytest_collection:whole_recording",
                "work_unit_index": 0,
                "source_clip_id": "whole_recording",
                "source_clip_index": 0,
                "frame_start": 0,
                "frame_stop": 4,
                "row_start": 0,
                "row_stop": 4,
            },
        ),
        source_producer_evidence=source_producer_evidence,
        source_producer_run_path=source_producer_run_path,
    )
    return manifest, source_receipt, source_path


def _source_manifest() -> dict[str, object]:
    return {
        "schema_id": "palette.subject_mask.source_fixture",
        "schema_version": 1,
        "run_id": "source_001",
    }


def _reference_receipt(
    *,
    kind: str,
    arrays: dict[str, np.ndarray],
    crop: dict[str, np.ndarray],
) -> dict[str, object]:
    schema = (
        RAW_SUBJECT_MASK_UINT8_SCHEMA_V1
        if kind == "raw_probability_uint8"
        else REFINED_SUBJECT_MASK_CORE_SCHEMA_V1
    )
    return build_reference_subject_mask_validation_receipt(
        kind=kind,
        source_run_path="subject_mask_shard_runs/source_001",
        source_manifest=_source_manifest(),
        schema=schema,
        arrays=arrays,
        dimensions=_dimensions(),
        components=_components(),
        threshold=0.5 if kind == "raw_probability_uint8" else None,
        source_crop_arrays=crop,
    )


@pytest.mark.parametrize(
    ("kind", "family", "payload"),
    (
        ("raw_probability_uint8", "subject_mask_runs", "mask_probs_roi"),
        ("refined_dense_core", "refined_subject_masks_runs", "masks_roi"),
    ),
)
def test_subject_mask_core_publication_round_trip(
    tmp_path: object,
    kind: str,
    family: str,
    payload: str,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    arrays = raw if kind == "raw_probability_uint8" else refined_with_crop
    run_id = f"{kind}_001"
    publication = publish_selector_ineligible_subject_mask_core_snapshot(
        arrays,
        source_crop_arrays=crop,  # type: ignore[arg-type]
        source_manifest={
            "schema_id": "palette.subject_mask.source_fixture",
            "schema_version": 1,
            "run_id": "source_001",
        },
        n_frames=4,
        components=_components(),
        destination=tmp_path / f"{kind}.zarr",  # type: ignore[operator]
        run_id=run_id,
        kind=kind,
        source_run_path="subject_mask_shard_runs/source_001",
        source_attributes={"source_crop_run": "crop_001"},
        created_by="pytest",
    )

    root = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=True
    )
    run = root[f"{family}/{run_id}"]
    assert run.attrs["stage_selector_eligible"] is False
    assert run.attrs["status"] == "complete"
    assert run.attrs[SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE] == publication.manifest
    assert set(run.array_keys()) | {
        f"metrics/{name}" for name in run["metrics"].array_keys()
    } == set(
        publication.plans.entries[index].rule.path
        for index in range(len(publication.plans.entries))
    )
    np.testing.assert_array_equal(run[payload][...], arrays[payload])
    parent = root[family]
    for selector in (
        "latest",
        "latest_complete",
        "latest_pending",
        "authoritative_run",
    ):
        assert parent.attrs.get(selector) is None


def test_coordinate_core_v4_binds_crop_raw_refined_and_worker_evidence(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop_arrays = refined_with_crop.pop("_crop")
    crop_manifest = _coordinate_crop_manifest()
    raw_source, raw_receipt, raw_source_path = _recording_documents(
        kind="raw_probability_uint8",
        arrays=raw,
    )
    raw_dependencies = build_subject_mask_core_coordinate_dependencies(
        kind="raw_probability_uint8",
        crop_run_path="crop_runs/crop_coordinate_shadow",
        crop_manifest=crop_manifest,
        source_crop_arrays=_crop_manifest_arrays(),
        source_run_path=raw_source_path,
        source_validation_receipt=raw_receipt,
        n_rois=4,
    )
    tampered_raw_receipt = deepcopy(raw_receipt)
    tampered_worker = tampered_raw_receipt["payload"]["producer_evidence"]["workers"][0]
    tampered_science = tampered_worker["scientific_identity"]
    tampered_science["payload"]["row_identity"]["arrays"]["instance_key"]["sha256"] = (
        "0" * 64
    )
    tampered_science["digest"] = canonical_json_sha256(tampered_science["payload"])
    tampered_attempt = tampered_worker["attempt"]
    tampered_attempt["payload"]["scientific_identity_digest"] = tampered_science[
        "digest"
    ]
    tampered_attempt["payload_digest"] = canonical_json_sha256(
        tampered_attempt["payload"]
    )
    tampered_semantic_receipt = tampered_worker["worker_receipt"]
    tampered_semantic_receipt["payload"]["scientific_identity_digest"] = (
        tampered_science["digest"]
    )
    tampered_semantic_receipt["payload"]["attempt_payload_digest"] = tampered_attempt[
        "payload_digest"
    ]
    tampered_semantic_receipt["payload_digest"] = canonical_json_sha256(
        tampered_semantic_receipt["payload"]
    )
    tampered_worker["scientific_identity_digest"] = tampered_science["digest"]
    tampered_worker["attempt_payload_digest"] = tampered_attempt["payload_digest"]
    tampered_worker["worker_receipt_payload_digest"] = tampered_semantic_receipt[
        "payload_digest"
    ]
    tampered_raw_receipt["payload_digest"] = canonical_json_sha256(
        tampered_raw_receipt["payload"]
    )
    with pytest.raises(ValueError, match="exact crop-v2 slice"):
        build_subject_mask_core_coordinate_dependencies(
            kind="raw_probability_uint8",
            crop_run_path="crop_runs/crop_coordinate_shadow",
            crop_manifest=crop_manifest,
            source_crop_arrays=_crop_manifest_arrays(),
            source_run_path=raw_source_path,
            source_validation_receipt=tampered_raw_receipt,
            n_rois=4,
        )
    raw_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        raw,
        source_crop_arrays=crop_arrays,  # type: ignore[arg-type]
        source_manifest=raw_source,
        n_frames=4,
        components=_components(),
        destination=tmp_path / "raw_coordinate_v4.zarr",  # type: ignore[operator]
        run_id="raw_coordinate_v4",
        kind="raw_probability_uint8",
        source_run_path=raw_source_path,
        threshold=0.5,
        validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        source_validation_receipt=raw_receipt,
        coordinate_dependencies=raw_dependencies,
    )
    assert (
        raw_publication.manifest["schema_version"]
        == SUBJECT_MASK_CORE_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION
    )
    assert validate_subject_mask_core_run_manifest(raw_publication.manifest) == ()
    stale_v3 = deepcopy(raw_publication.manifest)
    stale_v3["schema_version"] = 3
    assert "subject-mask core manifest envelope identity mismatch" in (
        validate_subject_mask_core_run_manifest(stale_v3)
    )

    refined_source, refined_receipt, refined_source_path = _recording_documents(
        kind="refined_dense_core",
        arrays=refined_with_crop,
        source_producer_evidence=raw_receipt["payload"]["producer_evidence"],
        source_producer_run_path=raw_source_path,
    )
    tampered_refined_receipt = deepcopy(refined_receipt)
    tampered_refined_receipt["payload"]["producer_evidence"]["source_producer_binding"][
        "digest"
    ] = ("0" * 64)
    tampered_refined_receipt["payload_digest"] = canonical_json_sha256(
        tampered_refined_receipt["payload"]
    )
    with pytest.raises(ValueError, match="invalid raw core"):
        build_subject_mask_core_coordinate_dependencies(
            kind="refined_dense_core",
            crop_run_path="crop_runs/crop_coordinate_shadow",
            crop_manifest=crop_manifest,
            source_crop_arrays=_crop_manifest_arrays(),
            source_run_path=refined_source_path,
            source_validation_receipt=tampered_refined_receipt,
            n_rois=4,
            raw_core_manifest=raw_publication.manifest,
        )
    refined_dependencies = build_subject_mask_core_coordinate_dependencies(
        kind="refined_dense_core",
        crop_run_path="crop_runs/crop_coordinate_shadow",
        crop_manifest=crop_manifest,
        source_crop_arrays=_crop_manifest_arrays(),
        source_run_path=refined_source_path,
        source_validation_receipt=refined_receipt,
        n_rois=4,
        raw_core_manifest=raw_publication.manifest,
    )
    refined_publication = publish_selector_ineligible_subject_mask_core_snapshot(
        refined_with_crop,
        source_crop_arrays=crop_arrays,  # type: ignore[arg-type]
        source_manifest=refined_source,
        n_frames=4,
        components=_components(),
        destination=tmp_path / "refined_coordinate_v4.zarr",  # type: ignore[operator]
        run_id="refined_coordinate_v4",
        kind="refined_dense_core",
        source_run_path=refined_source_path,
        validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        source_validation_receipt=refined_receipt,
        coordinate_dependencies=refined_dependencies,
    )
    refined_payload = refined_publication.manifest["payload"]
    assert refined_payload["coordinate_contract"]["document"] == (
        REFINED_SUBJECT_MASK_CORE_SCHEMA_V1.coordinate_contract_manifest()
    )
    assert (
        refined_payload["coordinate_dependencies"]["document"]["raw_core"][
            "manifest_payload_digest"
        ]
        == raw_publication.manifest["payload_digest"]
    )
    assert (
        refined_payload["coordinate_dependencies"]["document"]["assignment_keypoints"][
            "mode"
        ]
        == "not_used"
    )
    assert validate_subject_mask_core_run_manifest(refined_publication.manifest) == ()

    tampered = deepcopy(refined_publication.manifest)
    tampered["payload"]["coordinate_contract"]["document"]["surfaces"][0][
        "pixel_convention"
    ] = "wrong"
    tampered["payload"]["coordinate_contract"]["digest"] = canonical_json_sha256(
        tampered["payload"]["coordinate_contract"]["document"]
    )
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])
    assert "coordinate catalog differs from the frozen stage catalog" in (
        validate_subject_mask_core_run_manifest(tampered)
    )


def test_subject_mask_core_publication_rejects_crop_identity_mismatch(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    crop["instance_key"] = np.asarray([999, 102, 201, 301], dtype=np.uint64)

    with pytest.raises(ValueError, match="schema validation failed"):
        publish_selector_ineligible_subject_mask_core_snapshot(
            raw,
            source_crop_arrays=crop,  # type: ignore[arg-type]
            source_manifest={"schema_id": "fixture", "schema_version": 1},
            n_frames=4,
            components=_components(),
            destination=tmp_path / "raw.zarr",  # type: ignore[operator]
            run_id="raw_001",
            kind="raw_probability_uint8",
            source_run_path="subject_mask_shard_runs/source_001",
            created_by="pytest",
        )

    assert not (tmp_path / "raw.zarr").exists()  # type: ignore[operator]


def test_raw_publication_canonicalizes_one_ulp_probability_max(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    source_prob_max = raw["metrics/prob_max"].copy()
    source_prob_max[0, 0] = np.nextafter(
        source_prob_max[0, 0],
        np.float32(np.inf),
        dtype=np.float32,
    )
    raw["metrics/prob_max"] = source_prob_max

    publication = publish_selector_ineligible_subject_mask_core_snapshot(
        raw,
        source_crop_arrays=crop,  # type: ignore[arg-type]
        source_manifest={"schema_id": "fixture", "schema_version": 1},
        n_frames=4,
        components=_components(),
        destination=tmp_path / "raw.zarr",  # type: ignore[operator]
        run_id="raw_001",
        kind="raw_probability_uint8",
        source_run_path="subject_mask_shard_runs/source_001",
        created_by="pytest",
    )

    run = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=True
    )["subject_mask_runs/raw_001"]
    canonical = np.max(raw["mask_probs_roi"], axis=(2, 3)).astype(
        np.float32
    ) / np.float32(255.0)
    np.testing.assert_array_equal(run["metrics/prob_max"][...], canonical)
    receipt = publication.manifest["payload"]["write_receipt"][
        "derived_metric_canonicalization"
    ]
    assert receipt["source_mismatch_count"] == 1


def test_raw_publication_rejects_material_probability_max_drift(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    raw["metrics/prob_max"] = raw["metrics/prob_max"].copy()
    raw["metrics/prob_max"][0, 0] += np.float32(0.01)

    with pytest.raises(ValueError, match="differs materially"):
        publish_selector_ineligible_subject_mask_core_snapshot(
            raw,
            source_crop_arrays=crop,  # type: ignore[arg-type]
            source_manifest={"schema_id": "fixture", "schema_version": 1},
            n_frames=4,
            components=_components(),
            destination=tmp_path / "raw.zarr",  # type: ignore[operator]
            run_id="raw_001",
            kind="raw_probability_uint8",
            source_run_path="subject_mask_shard_runs/source_001",
            created_by="pytest",
        )


@pytest.mark.parametrize(
    ("kind", "family", "payload"),
    (
        ("raw_probability_uint8", "subject_mask_runs", "mask_probs_roi"),
        ("refined_dense_core", "refined_subject_masks_runs", "masks_roi"),
    ),
)
def test_subject_mask_production_streaming_publication_uses_validated_receipt(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    family: str,
    payload: str,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    arrays = raw if kind == "raw_probability_uint8" else refined_with_crop
    receipt = _reference_receipt(kind=kind, arrays=arrays, crop=crop)

    def forbid_full_postwrite_hash(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("production streaming performed a full postwrite hash")

    monkeypatch.setattr(core_publication, "_array_document", forbid_full_postwrite_hash)
    publication = publish_selector_ineligible_subject_mask_core_snapshot(
        arrays,
        source_crop_arrays=crop,
        source_manifest=_source_manifest(),
        n_frames=4,
        components=_components(),
        destination=tmp_path / f"streaming_{kind}.zarr",  # type: ignore[operator]
        run_id="streaming_001",
        kind=kind,
        source_run_path="subject_mask_shard_runs/source_001",
        created_by="pytest",
        validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        source_validation_receipt=receipt,
    )

    assert (
        publication.validation_mode
        is SubjectMaskCoreValidationMode.PRODUCTION_STREAMING
    )
    write_receipt = publication.manifest["payload"]["write_receipt"]
    assert write_receipt["logical_hash_timing"] == (
        "computed_during_required_publication_read_v1"
    )
    assert write_receipt["reopen_validation"] == (
        "metadata_plus_first_last_physical_row_band_samples_v1"
    )
    assert (
        publication.output_path
        / family
        / "streaming_001"
        / SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR
    ).is_file()
    assert (
        publication.manifest["payload"]["source"]["validation_receipt"]["storage"]
        == "strict_json_sidecar_v1"
    )
    run = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=True
    )[f"{family}/streaming_001"]
    np.testing.assert_array_equal(run[payload][...], arrays[payload])


def test_subject_mask_core_manifest_rejects_recomputed_nested_tampering(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    publication = publish_selector_ineligible_subject_mask_core_snapshot(
        raw,
        source_crop_arrays=crop,
        source_manifest=_source_manifest(),
        n_frames=4,
        components=_components(),
        destination=tmp_path / "tamper.zarr",  # type: ignore[operator]
        run_id="raw_001",
        kind="raw_probability_uint8",
        source_run_path="subject_mask_shard_runs/source_001",
    )
    tampered = deepcopy(publication.manifest)
    tampered["payload"]["storage_plan"]["arrays"][0][
        "access_unit_semantics"
    ] = "tampered"
    tampered["payload"]["source"]["unexpected"] = True
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    errors = validate_subject_mask_core_run_manifest(tampered)

    assert "subject-mask core storage plan differs from planner output" in errors
    assert "subject-mask core source binding is not exact" in errors


def test_subject_mask_production_streaming_requires_receipt(tmp_path: object) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    with pytest.raises(ValueError, match="requires a source-validation receipt"):
        publish_selector_ineligible_subject_mask_core_snapshot(
            raw,
            source_crop_arrays=crop,
            source_manifest=_source_manifest(),
            n_frames=4,
            components=_components(),
            destination=tmp_path / "missing_receipt.zarr",  # type: ignore[operator]
            run_id="raw_001",
            kind="raw_probability_uint8",
            source_run_path="subject_mask_shard_runs/source_001",
            validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        )


def test_subject_mask_production_streaming_rejects_source_changed_after_receipt(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    receipt = _reference_receipt(kind="raw_probability_uint8", arrays=raw, crop=crop)
    raw["instance_key"] = raw["instance_key"].copy()
    raw["instance_key"][0] = np.uint64(999)

    with pytest.raises(RuntimeError, match="differ from the validated source receipt"):
        publish_selector_ineligible_subject_mask_core_snapshot(
            raw,
            source_crop_arrays=crop,
            source_manifest=_source_manifest(),
            n_frames=4,
            components=_components(),
            destination=tmp_path / "mutated_source.zarr",  # type: ignore[operator]
            run_id="raw_001",
            kind="raw_probability_uint8",
            source_run_path="subject_mask_shard_runs/source_001",
            validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
            source_validation_receipt=receipt,
        )
    failed = zarr.open_group(
        str(tmp_path / "mutated_source.zarr"), mode="r", use_consolidated=False
    )["subject_mask_runs/raw_001"]
    assert failed.attrs["palette_run_completion_status"] == "failed"
    assert failed.attrs["stage_selector_eligible"] is False


def test_subject_mask_validation_receipt_rejects_recomputed_digest_gap() -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    receipt = _reference_receipt(kind="raw_probability_uint8", arrays=raw, crop=crop)
    tampered = deepcopy(receipt)
    coverage = tampered["payload"]["semantic_coverage"]
    coverage["units"][0]["stop_row"] = 3
    coverage["stop_row"] = 3
    coverage["units_digest"] = canonical_json_sha256(coverage["units"])
    tampered["payload_digest"] = canonical_json_sha256(tampered["payload"])

    with pytest.raises(ValueError, match="do not cover every ROI row"):
        validate_subject_mask_source_validation_receipt(
            tampered,
            kind="raw_probability_uint8",
            source_run_path="subject_mask_shard_runs/source_001",
            source_manifest=_source_manifest(),
            schema=RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
            arrays=raw,
            dimensions=_dimensions(),
            components=_components(),
            threshold=0.5,
        )


def test_subject_mask_validation_receipt_accepts_contiguous_worker_units(
    tmp_path: object,
) -> None:
    raw, refined_with_crop = _fixture()
    crop = refined_with_crop.pop("_crop")
    paths = tuple(
        binding.path
        for binding in RAW_SUBJECT_MASK_UINT8_SCHEMA_V1.bindings
        if binding.required
    )
    arrays = {path: raw[path] for path in paths}
    array_document = subject_mask_array_unit_document(arrays, paths, unit_rows=2)
    units = tuple(
        {
            "start_row": start,
            "stop_row": stop,
            "result": "valid",
            "validator_schema_id": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_ID,
            "validator_schema_version": SUBJECT_MASK_SOURCE_VALIDATOR_SCHEMA_VERSION,
            "evidence_digest": canonical_json_sha256(
                {"start_row": start, "stop_row": stop, "status": "valid"}
            ),
        }
        for start, stop in ((0, 2), (2, 4))
    )
    receipt = build_subject_mask_source_validation_receipt(
        kind="raw_probability_uint8",
        source_run_path="subject_mask_shard_runs/source_001",
        source_manifest=_source_manifest(),
        schema=RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
        arrays=arrays,
        dimensions=_dimensions(),
        components=_components(),
        threshold=0.5,
        array_document=array_document,
        semantic_units=units,
    )

    validated = validate_subject_mask_source_validation_receipt(
        receipt,
        kind="raw_probability_uint8",
        source_run_path="subject_mask_shard_runs/source_001",
        source_manifest=_source_manifest(),
        schema=RAW_SUBJECT_MASK_UINT8_SCHEMA_V1,
        arrays=arrays,
        dimensions=_dimensions(),
        components=_components(),
        threshold=0.5,
    )
    assert validated["payload"]["semantic_coverage"]["unit_count"] == 2

    publication = publish_selector_ineligible_subject_mask_core_snapshot(
        arrays,
        source_crop_arrays=crop,
        source_manifest=_source_manifest(),
        n_frames=4,
        components=_components(),
        destination=tmp_path / "unit_receipt.zarr",  # type: ignore[operator]
        run_id="raw_unit_receipt",
        kind="raw_probability_uint8",
        source_run_path="subject_mask_shard_runs/source_001",
        validation_mode=SubjectMaskCoreValidationMode.PRODUCTION_STREAMING,
        source_validation_receipt=receipt,
    )
    assert (
        publication.manifest["payload"]["logical_content"]["document"]["arrays"][
            "mask_probs_roi"
        ]["digest_algorithm"]
        == "sha256_c_contiguous_bytes_v1"
    )
