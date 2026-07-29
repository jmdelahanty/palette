from __future__ import annotations

import copy

from fisheye.shared.zarr.coordinate_manifest import (
    COORDINATE_CATALOG_ENVELOPE_SCHEMA_ID,
    COORDINATE_CATALOG_ENVELOPE_SCHEMA_VERSION,
    build_coordinate_catalog_envelope,
    validate_coordinate_catalog_envelope,
)
from fisheye.shared.zarr.detection_schema import CANONICAL_DETECTION_SCHEMA_V1
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


def _document() -> dict[str, object]:
    return CANONICAL_DETECTION_SCHEMA_V1.coordinate_contract_manifest()


def test_coordinate_catalog_envelope_binds_exact_frozen_document() -> None:
    document = _document()
    envelope = build_coordinate_catalog_envelope(document)

    assert envelope["schema_id"] == COORDINATE_CATALOG_ENVELOPE_SCHEMA_ID
    assert envelope["schema_version"] == (COORDINATE_CATALOG_ENVELOPE_SCHEMA_VERSION)
    assert envelope["digest"] == canonical_json_sha256(document)
    assert (
        validate_coordinate_catalog_envelope(
            envelope,
            expected_document=document,
        )
        == ()
    )


def test_recomputed_digest_cannot_hide_coordinate_catalog_tampering() -> None:
    document = _document()
    envelope = build_coordinate_catalog_envelope(document)
    tampered = copy.deepcopy(envelope)
    tampered["document"]["bindings"][0]["semantic_role"] = "sampled_spatial_surface"
    tampered["digest"] = canonical_json_sha256(tampered["document"])

    assert validate_coordinate_catalog_envelope(
        tampered,
        expected_document=document,
    ) == ("coordinate catalog differs from the frozen stage catalog",)


def test_coordinate_catalog_envelope_requires_exact_fields_and_strict_json() -> None:
    document = _document()
    extra = build_coordinate_catalog_envelope(document)
    extra["unexpected"] = True
    assert "unexpected field set" in " ".join(
        validate_coordinate_catalog_envelope(
            extra,
            expected_document=document,
        )
    )

    non_finite = build_coordinate_catalog_envelope(document)
    non_finite["document"]["unexpected"] = float("nan")
    assert "not strict JSON" in " ".join(
        validate_coordinate_catalog_envelope(
            non_finite,
            expected_document=document,
        )
    )
