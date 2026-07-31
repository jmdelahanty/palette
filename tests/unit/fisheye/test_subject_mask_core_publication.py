from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import zarr

import fisheye.shared.zarr.subject_mask_core_publication as core_publication
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR,
    SubjectMaskCoreValidationMode,
    publish_selector_ineligible_subject_mask_core_snapshot,
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
    build_reference_subject_mask_validation_receipt,
    build_subject_mask_source_validation_receipt,
    subject_mask_array_unit_document,
    validate_subject_mask_source_validation_receipt,
)


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
            [[0, 0, 8, 8], [1, 0, 8, 8], [0, 1, 8, 8], [1, 1, 8, 8]],
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
        publication.output_path / SUBJECT_MASK_CORE_SOURCE_VALIDATION_SIDECAR
    ).is_file()
    assert (
        publication.manifest["payload"]["source"]["validation_receipt"]["storage"]
        == "strict_json_sidecar_v1"
    )
    run = zarr.open_group(
        str(publication.output_path), mode="r", use_consolidated=True
    )[f"{family}/streaming_001"]
    np.testing.assert_array_equal(run[payload][...], arrays[payload])


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
