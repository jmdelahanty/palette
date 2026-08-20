"""Structural guards for compact, array-backed Zarr provenance.

Scientific provenance must remain readable, but observation-cardinality data
does not belong in Zarr attributes.  This module deliberately avoids a
universal byte limit: byte budgets are operational policy, whereas the
structural rule is scientific and stable.  Publishers provide the field names
whose values are authoritative arrays or recursively embedded source payloads;
the guard rejects those names anywhere in a proposed metadata record and
reports compact serialization measurements for tests and publication receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np


METADATA_CARDINALITY_PROFILE_SCHEMA_ID = "palette.zarr.metadata_cardinality_profile"
METADATA_CARDINALITY_PROFILE_SCHEMA_VERSION = 1


class MetadataCardinalityError(ValueError):
    """Raised when array-backed or recursive payload data enters metadata."""


@dataclass(frozen=True)
class MetadataCardinalityProfile:
    serialized_bytes: int
    mapping_entry_count: int
    sequence_element_count: int
    maximum_depth: int

    def to_json(self) -> dict[str, int | str]:
        return {
            "schema_id": METADATA_CARDINALITY_PROFILE_SCHEMA_ID,
            "schema_version": METADATA_CARDINALITY_PROFILE_SCHEMA_VERSION,
            "serialized_bytes": self.serialized_bytes,
            "mapping_entry_count": self.mapping_entry_count,
            "sequence_element_count": self.sequence_element_count,
            "maximum_depth": self.maximum_depth,
        }


def _walk(
    value: Any,
    *,
    path: str,
    depth: int,
    forbidden_fields: frozenset[str],
) -> tuple[int, int, int]:
    if isinstance(value, np.ndarray):
        raise MetadataCardinalityError(
            f"{path} is a NumPy array; observation values must be stored in a Zarr array."
        )
    if isinstance(value, Mapping):
        mapping_entries = len(value)
        sequence_elements = 0
        maximum_depth = depth
        for raw_key, item in value.items():
            if type(raw_key) is not str or not raw_key:
                raise MetadataCardinalityError(
                    f"{path} contains a non-canonical metadata field name."
                )
            if raw_key in forbidden_fields:
                raise MetadataCardinalityError(
                    f"{path}.{raw_key} is cardinality-scaled or recursively embedded metadata."
                )
            child = _walk(
                item,
                path=f"{path}.{raw_key}",
                depth=depth + 1,
                forbidden_fields=forbidden_fields,
            )
            mapping_entries += child[0]
            sequence_elements += child[1]
            maximum_depth = max(maximum_depth, child[2])
        return mapping_entries, sequence_elements, maximum_depth
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        mapping_entries = 0
        sequence_elements = len(value)
        maximum_depth = depth
        for index, item in enumerate(value):
            child = _walk(
                item,
                path=f"{path}[{index}]",
                depth=depth + 1,
                forbidden_fields=forbidden_fields,
            )
            mapping_entries += child[0]
            sequence_elements += child[1]
            maximum_depth = max(maximum_depth, child[2])
        return mapping_entries, sequence_elements, maximum_depth
    if value is None or type(value) in {str, int, float, bool}:
        return 0, 0, depth
    raise MetadataCardinalityError(
        f"{path} contains unsupported metadata value type {type(value).__name__}."
    )


def require_cardinality_independent_metadata(
    value: Mapping[str, Any],
    *,
    forbidden_fields: Sequence[str],
    label: str,
) -> MetadataCardinalityProfile:
    """Validate one strict JSON record and return boundedness measurements.

    ``forbidden_fields`` is schema-specific.  The generic implementation does
    not guess from list length because legitimate fixed-schema records (for
    example a 3x3 transform or a reason-code dictionary) are still useful,
    readable provenance.
    """

    if not isinstance(value, Mapping) or not value:
        raise MetadataCardinalityError(f"{label} must be one nonempty metadata object.")
    forbidden = frozenset(str(item) for item in forbidden_fields)
    mapping_entries, sequence_elements, maximum_depth = _walk(
        value,
        path=label,
        depth=0,
        forbidden_fields=forbidden,
    )
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MetadataCardinalityError(
            f"{label} must be strict finite JSON metadata."
        ) from exc
    return MetadataCardinalityProfile(
        serialized_bytes=len(encoded),
        mapping_entry_count=mapping_entries,
        sequence_element_count=sequence_elements,
        maximum_depth=maximum_depth,
    )


__all__ = [
    "METADATA_CARDINALITY_PROFILE_SCHEMA_ID",
    "METADATA_CARDINALITY_PROFILE_SCHEMA_VERSION",
    "MetadataCardinalityError",
    "MetadataCardinalityProfile",
    "require_cardinality_independent_metadata",
]
