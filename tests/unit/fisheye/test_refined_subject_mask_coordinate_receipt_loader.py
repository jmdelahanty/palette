from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

import fisheye.shared.refined_subject_mask_coordinate_publication as module
from fisheye.shared.refined_subject_mask_coordinate_publication import (
    BoundRefinedSubjectMaskCoordinateSurfaces,
    load_persisted_ineligible_refined_subject_mask_coordinate_surfaces,
    load_persisted_refined_subject_mask_coordinate_surfaces,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
)
from fisheye.shared.zarr.subject_mask_coordinate_validation_receipt import (
    REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
    SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
    build_subject_mask_coordinate_validation_receipt,
    validate_subject_mask_coordinate_validation_receipt,
)
from tests.unit.fisheye.test_refined_subject_mask_coordinate_publication import (
    _publish_activate,
    _refined_fixture,
)


_SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64


def test_ineligible_loader_does_not_require_production_activation_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    observed: dict[str, Any] = {}

    def load(*_args: Any, **kwargs: Any) -> Any:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(module, "_load_refined_subject_mask_coordinate_surfaces", load)
    result = load_persisted_ineligible_refined_subject_mask_coordinate_surfaces(
        object(),
        "refined_subject_masks_runs/successor",
    )
    assert result is sentinel
    assert observed["require_complete"] is True
    assert observed["require_activation_receipt"] is False
    assert observed["expected_selector_eligible"] is False


def _published(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any, Any]:
    root, parent, run, raw, snapshot = _refined_fixture(monkeypatch, fresh=True)
    loaded = _publish_activate(root, parent, run, snapshot)
    return root, run, raw, loaded


def _receipt(loaded: BoundRefinedSubjectMaskCoordinateSurfaces) -> dict[str, Any]:
    values = {
        "component_qc_inventory": loaded.component_qc_inventory,
        "context": loaded.context.context_record,
        "measurement_authority": loaded.measurement_authority,
        "refinement_authority": loaded.context.refinement_authority,
        "row_identity": loaded.context.row_identity,
        "scientific_manifest": loaded.scientific_manifest,
        "source_authority": loaded.context.source_authority,
        "surface_inventory": loaded.inventory,
        "temporal_authority": loaded.context.temporal_authority,
    }
    return build_subject_mask_coordinate_validation_receipt(
        kind=REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
        successor_run_path=loaded.context.run_path,
        source={
            "run_path": "refined_subject_masks_runs/source",
            "core_manifest_payload_digest": _SHA,
            "core_manifest_document_digest": _SHA_B,
            "logical_content_digest": _SHA_C,
        },
        source_validation={
            "schema_id": "palette.subject_mask.source_validation_receipt",
            "schema_version": 2,
            "payload_digest": _SHA_D,
            "document_sha256": _SHA_E,
            "semantic_unit_count": 1,
        },
        bundle_authority={"kind": "test_bundle", "document_digest": _SHA_F},
        coordinate_records={
            name: {
                "record_ref": value.record_ref,
                "record_sha256": value.record_sha256,
            }
            for name, value in values.items()
        },
        coordinate_record_names=tuple(values),
        payload_equivalence={
            "schema_id": "palette.coordinate_successor_payload_file_equivalence",
            "schema_version": 1,
            "receipt_digest": _SHA,
            "inventory_digest": _SHA_B,
            "payload_file_count": 0,
        },
        validator_identity={"package": "palette", "version": "receipt-test"},
    )


def test_no_receipt_keeps_the_existing_full_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    root, run, _raw, _loaded = _published(monkeypatch)
    calls = 0
    original = module._surface_evidence

    def scan(context: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(context)

    monkeypatch.setattr(module, "_surface_evidence", scan)
    loaded = load_persisted_refined_subject_mask_coordinate_surfaces(
        root,
        run.path,
    )
    assert type(loaded) is BoundRefinedSubjectMaskCoordinateSurfaces
    assert calls == 1
    assert SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE not in run.attrs


def test_valid_receipt_loader_does_not_index_dense_or_optional_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, run, _raw, loaded = _published(monkeypatch)
    context = module._load_refined_subject_mask_coordinate_context(
        root,
        run.path,
        require_complete=True,
        expected_selector_eligible=True,
    )
    receipt = _receipt(loaded)
    run.attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE] = receipt
    monkeypatch.setattr(
        module,
        "_load_refined_coordinate_receipt_authority",
        lambda *_args: (receipt, {}),
    )
    monkeypatch.setattr(
        module,
        "_load_refined_subject_mask_coordinate_context",
        lambda *_args, **_kwargs: context,
    )

    def fail_on_value_access(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("sealed dense scientific payload was read")

    monkeypatch.setattr(module, "_array", fail_on_value_access)
    result = load_persisted_refined_subject_mask_coordinate_surfaces(
        root,
        run.path,
    )
    assert type(result) is BoundRefinedSubjectMaskCoordinateSurfaces
    assert "masks_roi" in result.descriptors
    assert "components/subject_body/geometry/ellipse_params" in result.descriptors


@pytest.mark.parametrize(
    "attr_name",
    ("derived_mask_caches_stale", "metrics_stale", "contours_stale"),
)
def test_receipt_loader_rejects_stale_derived_state_before_dense_read(
    monkeypatch: pytest.MonkeyPatch,
    attr_name: str,
) -> None:
    root, run, _raw, loaded = _published(monkeypatch)
    context = module._load_refined_subject_mask_coordinate_context(
        root,
        run.path,
        require_complete=True,
        expected_selector_eligible=True,
    )
    receipt = _receipt(loaded)
    run.attrs[SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE] = receipt
    run.attrs[attr_name] = True
    monkeypatch.setattr(
        module,
        "_load_refined_subject_mask_coordinate_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        module,
        "_array",
        lambda *_args, **_kwargs: pytest.fail("dense read"),
    )

    with pytest.raises(
        module.RefinedSubjectMaskCoordinatePublicationError,
        match=f"fresh {attr_name}=False",
    ):
        load_persisted_refined_subject_mask_coordinate_surfaces(root, run.path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda receipt: receipt["payload"]["coordinate_records"].pop(
                "scientific_manifest"
            ),
            "coordinate validation receipt",
        ),
        (
            lambda receipt: receipt["payload"]["source"].update(
                logical_content_digest=_SHA_D
            ),
            "stale",
        ),
    ],
)
def test_stale_receipt_is_rejected_before_dense_read(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    root, run, _raw, loaded = _published(monkeypatch)
    receipt = _receipt(loaded)
    mutation(receipt)
    monkeypatch.setattr(module, "_array", lambda *_a, **_k: pytest.fail("dense read"))
    with pytest.raises(
        Exception,
        match=message,
    ):
        validate_subject_mask_coordinate_validation_receipt(
            receipt,
            expected_kind=REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
            expected_successor_run_path=run.path,
            expected_coordinate_record_names=receipt["payload"][
                "coordinate_records"
            ].keys(),
        )


def test_receipt_array_shape_and_dtype_tampering_fail_without_indexing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MetadataArray:
        path = "archive/refined_subject_masks_runs/r1/masks_roi"
        shape = (3, 2, 4, 4)
        dtype = np.dtype("uint8")

        def __getitem__(self, _key: object) -> object:
            raise AssertionError("array payload was indexed")

    node = MetadataArray()
    monkeypatch.setattr(module, "_receipt_node", lambda *_args, **_kwargs: node)
    payload = {
        "array_ref": "/archive/refined_subject_masks_runs/r1/masks_roi",
        "shape": [3, 2, 4, 4],
        "dtype": "|u1",
    }
    assert module._receipt_array_node(
        object(),
        run_path="refined_subject_masks_runs/r1",
        payload=payload,
        label="masks_roi",
    ) is not None
    for field, value in (("shape", [4, 2, 4, 4]), ("dtype", "<f4")):
        tampered = dict(payload)
        tampered[field] = value
        with pytest.raises(
            module.RefinedSubjectMaskCoordinatePublicationError,
            match="differs",
        ):
            module._receipt_array_node(
                object(),
                run_path="refined_subject_masks_runs/r1",
                payload=tampered,
                label="masks_roi",
            )


def _authority_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    root, run, _raw, loaded = _published(monkeypatch)
    receipt = _receipt(loaded)
    source_validation = {
        "schema_id": "palette.subject_mask.source_validation_receipt",
        "schema_version": 2,
        "payload_digest": _SHA_D,
        "document_sha256": _SHA_E,
        "semantic_unit_count": 3,
    }
    target_payload = {
        "run_id": run.path.split("/", 1)[1],
        "stage_family": "refined_subject_masks_runs",
        "kind": "refined_dense_core",
        "coordinate_contract": "canonical_v2",
        "logical_content": {"digest": _SHA_C},
        "source": {"validation_receipt": source_validation},
    }
    target_manifest = {
        "schema_id": "palette.subject_mask.core_run_manifest",
        "schema_version": 4,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(target_payload),
        "payload": target_payload,
    }
    run.attrs[SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE] = target_manifest
    source = {
        "run_path": run.path,
        "core_manifest_payload_digest": target_manifest["payload_digest"],
        "core_manifest_document_digest": canonical_json_sha256(target_manifest),
        "logical_content_digest": _SHA_C,
    }
    receipt["payload"]["source"] = source
    receipt["payload"]["source_validation"] = source_validation
    raw_bundle_member = {
        "role": "raw",
        "family": "subject_mask_runs",
        "run_id": "raw_source",
        "run_path": "subject_mask_runs/raw_source",
        "manifest_schema_id": "palette.subject_mask.core_run_manifest",
        "manifest_schema_version": 5,
        "manifest_payload_digest": _SHA,
        "manifest_document_digest": _SHA_B,
        "logical_content_digest": _SHA_C,
    }
    bundle_manifest = {
        "schema_id": "palette.subject_mask.bundle_manifest",
        "schema_version": 3,
        "payload": {
            "members": {"raw": raw_bundle_member}
        },
    }
    receipt["payload"]["bundle_authority"] = {
        "kind": "inactive_subject_mask_bundle_v3",
        "document_digest": canonical_json_sha256(bundle_manifest),
    }
    raw_authority = {
        "payload": {
            "kind": "raw_subject_mask_coordinate_successor",
            "source": {
                key: raw_bundle_member[key]
                for key in (
                    "family",
                    "run_path",
                    "manifest_schema_id",
                    "manifest_schema_version",
                    "manifest_payload_digest",
                    "manifest_document_digest",
                    "logical_content_digest",
                )
            },
            "successor": {"run_path": "subject_mask_runs/raw_successor"},
        }
    }
    authority_records = copy.deepcopy(receipt["payload"]["coordinate_records"])
    authority_records["coordinate_validation_receipt"] = {
        "record_ref": f"{run.path}@{SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE}",
        "record_sha256": _SHA_F,
    }
    authority = {
        "payload": {
            "source": {
                "run_path": run.path,
                "manifest_payload_digest": target_manifest["payload_digest"],
                "manifest_document_digest": canonical_json_sha256(target_manifest),
                "logical_content_digest": _SHA_C,
            },
            "successor": {"run_path": run.path},
            "coordinate_records": authority_records,
            "source_authority": {
                "kind": "inactive_subject_mask_bundle_v3_plus_raw_successor",
                "record": {
                    "bundle_manifest": bundle_manifest,
                    "raw_successor_authority": raw_authority,
                },
            },
            "payload_equivalence": {
                "source_logical_content_digest": _SHA_C,
                "payload_file_equivalence": receipt["payload"][
                    "payload_equivalence"
                ],
            },
        }
    }
    monkeypatch.setattr(module, "validate_subject_mask_core_run_manifest", lambda _value: ())
    monkeypatch.setattr(module, "validate_subject_mask_bundle_manifest", lambda _value: ())
    monkeypatch.setattr(module, "validate_coordinate_successor_authority", lambda *_a, **_k: ())
    monkeypatch.setattr(
        module,
        "load_coordinate_successor_authority",
        lambda *_a, **_k: authority,
    )
    monkeypatch.setattr(
        module,
        "load_subject_mask_coordinate_validation_receipt",
        lambda *_a, **_k: receipt,
    )
    return root, run, receipt, authority


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt, authority: receipt["payload"]["source"].update(
            logical_content_digest=_SHA_D
        ),
        lambda receipt, authority: receipt["payload"]["source_validation"].update(
            payload_digest=_SHA
        ),
        lambda receipt, authority: receipt["payload"]["bundle_authority"].update(
            kind="wrong_bundle"
        ),
        lambda receipt, authority: authority["payload"]["coordinate_records"][
            "context"
        ].update(record_sha256=_SHA_F),
    ],
)
def test_authority_source_bundle_and_record_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    root, run, receipt, authority = _authority_fixture(monkeypatch)
    mutation(receipt, authority)
    with pytest.raises(
        module.RefinedSubjectMaskCoordinatePublicationError,
        match="(source|validation|bundle|authority|record)",
    ):
        module._load_refined_coordinate_receipt_authority(root, run.path)
