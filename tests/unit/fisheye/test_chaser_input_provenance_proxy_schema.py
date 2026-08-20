from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    select_chaser_input_provenance_proxy,
)
from fisheye.shared.zarr.chaser_input_provenance_proxy_schema import (
    ChaserInputProvenanceProxySchemaError,
    build_publication_manifest,
    encode_reason_codes,
    validate_proxy_result,
    validate_publication_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from tests.unit.fisheye.test_chaser_input_provenance_proxy import _source


def _result():
    return select_chaser_input_provenance_proxy(_source())


def test_schema_preserves_three_axes_and_encodes_reason_arrays() -> None:
    result = _result()
    dimensions, arrays, record = validate_proxy_result(
        result, revalidate_source=select_chaser_input_provenance_proxy
    )
    encoded = encode_reason_codes(result)

    assert dimensions.as_manifest() == {
        "frame": 3,
        "candidate": 5,
        "chaser": 2,
        "frame_boundary": 4,
    }
    assert set(arrays) == set(encoded)
    assert encoded["candidate_reason_code"].dtype == np.dtype("uint8")
    assert encoded["selection_reason_code"].dtype == np.dtype("uint8")
    assert all(
        value.dtype.kind not in {"O", "U", "S"} for value in encoded.values()
    )
    assert record["behavioral_denominator"] == "unique_input_acquisition_frames"
    assert record["physical_presentation_verified"] is False


def test_manifest_binds_projection_source_dimensions_and_encoded_arrays() -> None:
    result = _result()
    arrays = encode_reason_codes(result)
    manifest = build_publication_manifest(result)

    normalized = validate_publication_manifest(manifest, arrays)

    assert normalized["selector_eligible"] is False
    assert normalized["selection"] == "none"
    assert normalized["dimensions"]["candidate"] == 5
    assert normalized["source"]["source_verification_digest"] == "b" * 64
    assert len(normalized["array_declarations"]) == len(arrays)


def test_selected_lineage_must_come_from_one_exact_candidate() -> None:
    result = _result()
    changed = result.selected_timestamp_ns_session.copy()
    changed[0] += 1

    with pytest.raises(ChaserInputProvenanceProxySchemaError, match="selected candidate"):
        validate_proxy_result(
            replace(result, selected_timestamp_ns_session=changed)
        )


def test_rehashed_physical_presentation_claim_is_rejected() -> None:
    result = _result()
    record = dict(result.acquisition_projection_record)
    record["physical_presentation_verified"] = True

    with pytest.raises(
        ChaserInputProvenanceProxySchemaError,
        match="physical_presentation_verified",
    ):
        validate_proxy_result(
            replace(
                result,
                acquisition_projection_record=record,
                acquisition_projection_record_sha256=canonical_json_sha256(record),
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("denominator_policy", "native_samples", "denominator"),
        ("no_carry_policy", "carry_forward", "no-carry"),
        (
            "reason_code_registry",
            {"encoding": "uint8", "codes": {}},
            "reason-code",
        ),
    ],
)
def test_manifest_rejects_rehashed_semantic_tampering(
    field: str, value: object, message: str
) -> None:
    manifest = build_publication_manifest(_result())
    manifest[field] = value

    with pytest.raises(ChaserInputProvenanceProxySchemaError, match=message):
        validate_publication_manifest(manifest)


def test_manifest_rejects_array_content_tampering() -> None:
    result = _result()
    manifest = build_publication_manifest(result)
    arrays = encode_reason_codes(result)
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["selected_stimulus_frame_num"][0] += 1

    with pytest.raises(ChaserInputProvenanceProxySchemaError, match="content digest"):
        validate_publication_manifest(manifest, changed)
