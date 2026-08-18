from __future__ import annotations

import copy

import pytest

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_coordinate_validation_receipt import (
    RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
    SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
    SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE,
    SubjectMaskCoordinateValidationReceiptError,
    build_subject_mask_coordinate_validation_receipt,
    load_subject_mask_coordinate_validation_receipt,
    stamp_subject_mask_coordinate_validation_receipt,
    validate_subject_mask_coordinate_validation_receipt,
)


_SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64


def _inputs(
    *,
    semantic_unit_count: int = 3,
    payload_file_count: int = 2,
    **overrides,
):
    values = {
        "kind": RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
        "successor_run_path": "subject_mask_runs/successor_v1",
        "source": {
            "run_path": "subject_mask_runs/source_v1",
            "core_manifest_payload_digest": _SHA,
            "core_manifest_document_digest": _SHA_B,
            "logical_content_digest": _SHA_C,
        },
        "source_validation": {
            "schema_id": "palette.subject_mask.source_validation_receipt",
            "schema_version": 2,
            "payload_digest": _SHA_D,
            "document_sha256": _SHA_E,
            "semantic_unit_count": semantic_unit_count,
        },
        "bundle_authority": {
            "kind": "subject_mask_bundle_authority_v1",
            "document_digest": _SHA_F,
        },
        "coordinate_records": {
            "coordinate_context": {
                "record_ref": "/archive/recording.zarr/subject_mask_runs/successor_v1/zarr.json@coordinate_context",
                "record_sha256": _SHA,
            },
            "coordinate_derivation": {
                "record_ref": "/archive/recording.zarr/subject_mask_runs/successor_v1/zarr.json@coordinate_derivation",
                "record_sha256": _SHA_B,
            },
        },
        "coordinate_record_names": ("coordinate_context", "coordinate_derivation"),
        "payload_equivalence": {
            "schema_id": "palette.coordinate_successor_payload_file_equivalence",
            "schema_version": 1,
            "receipt_digest": _SHA_C,
            "inventory_digest": _SHA_D,
            "payload_file_count": payload_file_count,
        },
        "validator_identity": {"package": "palette", "version": "test-1"},
    }
    values.update(overrides)
    return values


def _build(**overrides):
    values = _inputs()
    values.update(overrides)
    return build_subject_mask_coordinate_validation_receipt(**values)


class _Attrs(dict):
    pass


class _FakeRun:
    path = "subject_mask_runs/successor_v1"

    def __init__(self):
        self.attrs = _Attrs()

    def __getitem__(self, key):  # pragma: no cover - receipt must never use it
        raise AssertionError(f"scientific payload was accessed: {key!r}")


def test_build_stamp_load_roundtrip_is_metadata_only():
    run = _FakeRun()
    receipt = _build()
    stamped = stamp_subject_mask_coordinate_validation_receipt(
        run,
        receipt,
        expected_kind=RAW_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
        expected_record_names=receipt["payload"]["coordinate_records"].keys(),
    )
    assert stamped == receipt
    assert run.attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE] == receipt
    assert run.attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE] == (
        canonical_json_sha256(receipt)
    )
    assert load_subject_mask_coordinate_validation_receipt(
        run,
        expected_successor_run_path="subject_mask_runs/successor_v1",
    ) == receipt


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_id", "wrong.schema"),
        ("schema_version", 2),
        ("digest_algorithm", "wrong"),
        ("payload", {}),
    ],
)
def test_tampered_envelope_is_rejected(field, replacement):
    receipt = _build()
    tampered = copy.deepcopy(receipt)
    tampered[field] = replacement
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        validate_subject_mask_coordinate_validation_receipt(tampered)


def test_stale_payload_digest_is_rejected():
    receipt = _build()
    tampered = copy.deepcopy(receipt)
    tampered["payload"]["validation_policy"] = "other"
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        validate_subject_mask_coordinate_validation_receipt(tampered)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("source", "core_manifest_payload_digest", _SHA_D),
        ("source_validation", "payload_digest", _SHA),
        ("bundle_authority", "document_digest", _SHA),
        ("payload_equivalence", "inventory_digest", _SHA),
        ("coordinate_records", "coordinate_context", None),
    ],
)
def test_nested_binding_tampering_is_rejected(section, field, replacement):
    receipt = _build()
    tampered = copy.deepcopy(receipt)
    if section == "coordinate_records":
        tampered["payload"][section][field]["record_ref"] = "not-an-absolute-ref"
    else:
        tampered["payload"][section][field] = replacement
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        validate_subject_mask_coordinate_validation_receipt(tampered)


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "refined_subject_mask"},
        {"successor_run_path": "refined_subject_masks_runs/successor_v1"},
        {"coordinate_record_names": ("coordinate_context",)},
        {"source_validation": {"schema_id": "bad"}},
    ],
)
def test_kind_path_and_closed_record_bindings_are_strict(overrides):
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        _build(**overrides)


def test_wrong_field_sets_and_invalid_record_pointer_are_rejected():
    values = _inputs()
    values["source"] = {**values["source"], "unexpected": True}
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        build_subject_mask_coordinate_validation_receipt(**values)

    values = _inputs()
    values["payload_equivalence"] = {
        **values["payload_equivalence"],
        "unexpected": True,
    }
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        build_subject_mask_coordinate_validation_receipt(**values)

    values = _inputs()
    values["coordinate_records"] = {
        "coordinate_context": {
            "record_ref": "relative/path@record",
            "record_sha256": _SHA,
        },
        "coordinate_derivation": {
            "record_ref": "/archive/recording.zarr@record",
            "record_sha256": _SHA_B,
        },
    }
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        build_subject_mask_coordinate_validation_receipt(**values)


def test_stale_companion_digest_is_rejected_on_load():
    run = _FakeRun()
    receipt = _build()
    stamp_subject_mask_coordinate_validation_receipt(run, receipt)
    run.attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_DIGEST_ATTRIBUTE] = _SHA
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        load_subject_mask_coordinate_validation_receipt(run)


def test_stamp_is_idempotent_but_never_overwrites_occupied_evidence():
    run = _FakeRun()
    receipt = _build()
    assert stamp_subject_mask_coordinate_validation_receipt(run, receipt) == receipt
    assert stamp_subject_mask_coordinate_validation_receipt(run, receipt) == receipt

    replacement = _build(
        validator_identity={"package": "palette", "version": "different"}
    )
    with pytest.raises(
        SubjectMaskCoordinateValidationReceiptError,
        match="already occupied",
    ):
        stamp_subject_mask_coordinate_validation_receipt(run, replacement)


def test_zero_semantic_units_and_payload_files_are_valid():
    receipt = _build(
        source_validation={
            **_inputs()["source_validation"],
            "semantic_unit_count": 0,
        },
        payload_equivalence={
            **_inputs()["payload_equivalence"],
            "payload_file_count": 0,
        },
    )
    assert receipt["payload"]["source_validation"]["semantic_unit_count"] == 0
    assert receipt["payload"]["payload_equivalence"]["payload_file_count"] == 0


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("source_validation", "semantic_unit_count"),
        ("payload_equivalence", "payload_file_count"),
    ],
)
def test_boolean_counts_are_not_integers(section, field):
    values = _inputs()
    values[section] = {**values[section], field: True}
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        build_subject_mask_coordinate_validation_receipt(**values)


def test_validator_identity_requires_package_version_or_commit():
    values = _inputs(validator_identity={"package": "palette"})
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        build_subject_mask_coordinate_validation_receipt(**values)

    receipt = _build(validator_identity={"commit": "abc123"})
    assert receipt["payload"]["validator_identity"] == {"commit": "abc123"}


def test_expected_kind_path_and_record_names_are_checked_on_load():
    receipt = _build()
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        validate_subject_mask_coordinate_validation_receipt(
            receipt, expected_kind="refined_subject_mask"
        )
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        validate_subject_mask_coordinate_validation_receipt(
            receipt,
            expected_successor_run_path="subject_mask_runs/other",
        )
    with pytest.raises(SubjectMaskCoordinateValidationReceiptError):
        validate_subject_mask_coordinate_validation_receipt(
            receipt,
            expected_record_names=("coordinate_context",),
        )
