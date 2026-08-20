"""Prepare one chaser input-provenance proxy for immutable publication."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.analysis_workflows.chaser_input_provenance_proxy import (
    ChaserInputProvenanceProxyResult,
    select_chaser_input_provenance_proxy,
)
from fisheye.shared.zarr.chaser_input_provenance_proxy_schema import (
    ChaserInputProvenanceProxyDimensions,
    build_publication_manifest,
    encode_reason_codes,
    validate_proxy_result,
    validate_publication_manifest,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


PREPARED_PROXY_SCHEMA_ID = (
    "palette.analysis.chaser_input_provenance_proxy.prepared_candidate"
)
PREPARED_PROXY_SCHEMA_VERSION = 1


class ChaserInputProvenanceProxyStorageError(ValueError):
    """Raised when proxy evidence cannot form one immutable payload."""


def _fail(message: str) -> None:
    raise ChaserInputProvenanceProxyStorageError(message)


def _readonly_arrays(values: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        array = np.array(value, copy=True, order="C")
        if array.dtype.hasobject or array.dtype.kind in {"U", "S"}:
            _fail(f"Prepared proxy array {name!r} is object/string typed.")
        array.setflags(write=False)
        result[name] = array
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class PreparedChaserInputProvenanceProxy:
    """Typed arrays and their complete immutable publication manifest."""

    dimensions: ChaserInputProvenanceProxyDimensions
    arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    @property
    def payload_digest(self) -> str:
        return canonical_json_sha256(dict(self.manifest))


def prepare_chaser_input_provenance_proxy(
    result: ChaserInputProvenanceProxyResult,
) -> PreparedChaserInputProvenanceProxy:
    """Revalidate the sealed source and prepare numeric row evidence."""

    dimensions, _arrays, _record = validate_proxy_result(
        result,
        revalidate_source=select_chaser_input_provenance_proxy,
    )
    arrays = _readonly_arrays(encode_reason_codes(result))
    manifest = validate_publication_manifest(
        build_publication_manifest(result),
        arrays,
    )
    prepared_manifest = {
        **manifest,
        "prepared_candidate": {
            "schema_id": PREPARED_PROXY_SCHEMA_ID,
            "schema_version": PREPARED_PROXY_SCHEMA_VERSION,
            "candidate_state": "validated_prepared_selector_ineligible",
        },
    }
    return PreparedChaserInputProvenanceProxy(
        dimensions=dimensions,
        arrays=arrays,
        manifest=MappingProxyType(prepared_manifest),
    )


def validate_prepared_chaser_input_provenance_proxy(
    prepared: PreparedChaserInputProvenanceProxy,
) -> dict[str, Any]:
    """Recheck every array, count, policy, and manifest content digest."""

    if not isinstance(prepared, PreparedChaserInputProvenanceProxy):
        _fail("prepared must be one PreparedChaserInputProvenanceProxy.")
    candidate = prepared.manifest.get("prepared_candidate")
    if candidate != {
        "schema_id": PREPARED_PROXY_SCHEMA_ID,
        "schema_version": PREPARED_PROXY_SCHEMA_VERSION,
        "candidate_state": "validated_prepared_selector_ineligible",
    }:
        _fail("Prepared proxy candidate identity is missing or invalid.")
    publication_manifest = {
        key: value
        for key, value in prepared.manifest.items()
        if key != "prepared_candidate"
    }
    normalized = validate_publication_manifest(
        publication_manifest,
        prepared.arrays,
    )
    expected_dimensions = prepared.dimensions.as_manifest()
    if normalized.get("dimensions") != expected_dimensions:
        _fail("Prepared dimensions differ from the publication manifest.")
    if any(np.asarray(value).flags.writeable for value in prepared.arrays.values()):
        _fail("Prepared arrays must remain read-only snapshots.")
    return {
        "schema_id": "palette.chaser_input_provenance_proxy.validation_receipt",
        "schema_version": 1,
        "payload_digest": prepared.payload_digest,
        "dimensions": expected_dimensions,
        "array_count": len(prepared.arrays),
        "selector_eligible": False,
        "selection": "none",
    }


__all__ = [
    "PREPARED_PROXY_SCHEMA_ID",
    "PREPARED_PROXY_SCHEMA_VERSION",
    "ChaserInputProvenanceProxyStorageError",
    "PreparedChaserInputProvenanceProxy",
    "prepare_chaser_input_provenance_proxy",
    "validate_prepared_chaser_input_provenance_proxy",
]
