"""Versioned per-row source signatures for incremental materialization.

``instance_key`` identifies an observation; it does not prove that the
observation's geometry, pixels, or reviewed upstream payload are unchanged.
This module supplies the separate, stage-specific signature used to make that
reuse decision.  Callers may process one bounded row batch at a time and write
the returned ``uint8[N, 32]`` values directly to Zarr.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


ROW_SOURCE_SIGNATURE_ARRAY = "source_row_signature"
ROW_SOURCE_SIGNATURE_SCHEMA_ID = "palette.row_source_signature"
ROW_SOURCE_SIGNATURE_SCHEMA_VERSION = 1
ROW_SOURCE_SIGNATURE_ALGORITHM = "sha256"
ROW_SOURCE_SIGNATURE_WIDTH_BYTES = hashlib.sha256().digest_size
ROW_SOURCE_SIGNATURE_CANONICALIZATION = (
    "canonical_json_spec_and_little_endian_numeric_rows_v1"
)
ROW_SOURCE_SIGNATURE_BASIS_CONTENT = "content"
ROW_SOURCE_SIGNATURE_BASIS_REVISION = "revision"
ROW_SOURCE_SIGNATURE_BASIS_CONTENT_AND_REVISION = "content_and_revision"


class RowSourceSignatureError(ValueError):
    """Raised when row inputs cannot satisfy the reuse-signature contract."""


@dataclass(frozen=True)
class RowSourceComponentSpec:
    """Canonical description of one row-aligned signature component."""

    name: str
    basis: str
    dtype: str
    trailing_shape: tuple[int, ...]
    revision_authority: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "basis": self.basis,
            "dtype": self.dtype,
            "trailing_shape": list(self.trailing_shape),
        }
        if self.revision_authority is not None:
            payload["revision_authority"] = self.revision_authority
        return payload


@dataclass(frozen=True)
class RowSourceSignatureSpec:
    """The exact schema/context under which a row digest was produced."""

    stage: str
    basis: str
    components: tuple[RowSourceComponentSpec, ...]
    compatibility_context: Mapping[str, Any]
    canonical_json: str
    spec_digest: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)

    def to_attrs(self, *, prefix: str = "source_row_signature_") -> dict[str, Any]:
        """Return JSON-safe Zarr attrs that make comparisons fail closed."""

        return {
            f"{prefix}schema_id": ROW_SOURCE_SIGNATURE_SCHEMA_ID,
            f"{prefix}schema_version": ROW_SOURCE_SIGNATURE_SCHEMA_VERSION,
            f"{prefix}algorithm": ROW_SOURCE_SIGNATURE_ALGORITHM,
            f"{prefix}width_bytes": ROW_SOURCE_SIGNATURE_WIDTH_BYTES,
            f"{prefix}canonicalization": ROW_SOURCE_SIGNATURE_CANONICALIZATION,
            f"{prefix}stage": self.stage,
            f"{prefix}basis": self.basis,
            f"{prefix}spec_digest": self.spec_digest,
            f"{prefix}spec": self.to_dict(),
        }


@dataclass(frozen=True)
class RowSourceSignatureBatch:
    """One bounded batch of row signatures plus its comparison specification."""

    signatures: np.ndarray
    spec: RowSourceSignatureSpec


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _normalize_json(value: Any, *, path: str = "compatibility_context") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _normalize_json(value.item(), path=path)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise RowSourceSignatureError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key:
                raise RowSourceSignatureError(f"{path} contains an empty key.")
            if key in normalized:
                raise RowSourceSignatureError(f"{path} contains duplicate key {key!r}.")
            normalized[key] = _normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, path=f"{path}[]") for item in value]
    raise RowSourceSignatureError(
        f"{path} contains unsupported value type {type(value).__name__}."
    )


def _canonical_content_array(
    values: np.ndarray,
    *,
    name: str,
    row_count: int,
) -> tuple[np.ndarray, str]:
    arr = np.asarray(values)
    if arr.ndim < 1 or int(arr.shape[0]) != row_count:
        raise RowSourceSignatureError(
            f"Content component {name!r} must have leading row count {row_count}."
        )
    if arr.dtype.kind not in "biuf":
        raise RowSourceSignatureError(
            f"Content component {name!r} must use a fixed-width bool/integer/float dtype."
        )
    if arr.dtype.kind in "iuf" and int(arr.dtype.itemsize) not in (1, 2, 4, 8):
        raise RowSourceSignatureError(
            f"Content component {name!r} uses unsupported {arr.dtype.itemsize}-byte values."
        )

    if arr.dtype.kind == "b":
        canonical = np.ascontiguousarray(arr, dtype=np.uint8)
        dtype_text = "|b1"
    else:
        little_dtype = arr.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(arr, dtype=little_dtype)
        dtype_text = little_dtype.str
        if canonical.dtype.kind == "f":
            canonical = canonical.copy()
            canonical[canonical == 0] = 0.0
            canonical[np.isnan(canonical)] = np.nan
    return canonical, dtype_text


def _canonical_revision_array(
    values: np.ndarray,
    *,
    name: str,
    row_count: int,
) -> tuple[np.ndarray, str]:
    arr = np.asarray(values)
    if arr.ndim < 1 or int(arr.shape[0]) != row_count:
        raise RowSourceSignatureError(
            f"Revision component {name!r} must have leading row count {row_count}."
        )
    if arr.dtype.kind not in "iu":
        raise RowSourceSignatureError(
            f"Revision component {name!r} must use an integer dtype."
        )
    if np.any(arr < 0):
        raise RowSourceSignatureError(
            f"Revision component {name!r} contains a negative revision."
        )
    return np.ascontiguousarray(arr, dtype="<u8"), "<u8"


def _component_name(raw_name: object) -> str:
    name = str(raw_name).strip()
    if not name:
        raise RowSourceSignatureError("Signature component names must be non-empty.")
    return name


def build_row_source_signatures(
    *,
    stage: str,
    instance_keys: np.ndarray,
    content_components: Mapping[str, np.ndarray] | None = None,
    revision_components: Mapping[str, np.ndarray] | None = None,
    revision_authorities: Mapping[str, str] | None = None,
    compatibility_context: Mapping[str, Any] | None = None,
) -> RowSourceSignatureBatch:
    """Build deterministic per-row source signatures for one bounded batch.

    ``content_components`` contain source values whose changes invalidate the
    row. ``revision_components`` may substitute for content only when the named
    authority guarantees that every relevant accepted edit increments that
    nonnegative row revision. Global stage inputs (model/config/schema/pixel
    identity) belong in ``compatibility_context``.
    """

    stage_name = str(stage).strip()
    if not stage_name:
        raise RowSourceSignatureError("stage must be non-empty.")

    raw_keys = np.asarray(instance_keys)
    if raw_keys.dtype.kind not in "iu":
        raise RowSourceSignatureError("instance_key values must use an integer dtype.")
    if raw_keys.dtype.kind == "i" and np.any(raw_keys < 0):
        raise RowSourceSignatureError("instance_key values must be nonnegative.")
    keys = np.asarray(raw_keys, dtype=np.uint64).reshape(-1)
    row_count = int(keys.shape[0])
    if int(np.unique(keys).shape[0]) != row_count:
        raise RowSourceSignatureError(
            "instance_key values must be unique within a batch."
        )
    canonical_keys = np.ascontiguousarray(keys, dtype="<u8")

    content = content_components or {}
    revisions = revision_components or {}
    if not content and not revisions:
        raise RowSourceSignatureError(
            "At least one content or revision component is required."
        )

    normalized_authorities: dict[str, str] = {}
    for raw_name, raw_authority in (revision_authorities or {}).items():
        name = _component_name(raw_name)
        authority = str(raw_authority).strip()
        if not authority:
            raise RowSourceSignatureError(
                f"Revision component {name!r} has an empty authority."
            )
        if name in normalized_authorities:
            raise RowSourceSignatureError(
                f"Revision authority for component {name!r} is declared more than once."
            )
        normalized_authorities[name] = authority

    normalized_inputs: list[tuple[RowSourceComponentSpec, np.ndarray]] = []
    seen_names: set[str] = set()
    seen_revision_names: set[str] = set()
    for basis, values_by_name in (
        (ROW_SOURCE_SIGNATURE_BASIS_CONTENT, content),
        (ROW_SOURCE_SIGNATURE_BASIS_REVISION, revisions),
    ):
        for raw_name, values in values_by_name.items():
            name = _component_name(raw_name)
            if name in seen_names:
                raise RowSourceSignatureError(
                    f"Signature component {name!r} is declared more than once."
                )
            seen_names.add(name)
            if basis == ROW_SOURCE_SIGNATURE_BASIS_CONTENT:
                canonical, dtype_text = _canonical_content_array(
                    values,
                    name=name,
                    row_count=row_count,
                )
            else:
                seen_revision_names.add(name)
                if name not in normalized_authorities:
                    raise RowSourceSignatureError(
                        f"Revision component {name!r} requires a named revision authority."
                    )
                canonical, dtype_text = _canonical_revision_array(
                    values,
                    name=name,
                    row_count=row_count,
                )
            spec = RowSourceComponentSpec(
                name=name,
                basis=basis,
                dtype=dtype_text,
                trailing_shape=tuple(int(value) for value in canonical.shape[1:]),
                revision_authority=(
                    normalized_authorities[name]
                    if basis == ROW_SOURCE_SIGNATURE_BASIS_REVISION
                    else None
                ),
            )
            normalized_inputs.append((spec, canonical))

    unused_authorities = sorted(normalized_authorities.keys() - seen_revision_names)
    if unused_authorities:
        raise RowSourceSignatureError(
            "Revision authorities were supplied for undeclared components: "
            f"{unused_authorities}."
        )

    normalized_inputs.sort(key=lambda item: (item[0].name, item[0].basis))
    if content and revisions:
        signature_basis = ROW_SOURCE_SIGNATURE_BASIS_CONTENT_AND_REVISION
    elif content:
        signature_basis = ROW_SOURCE_SIGNATURE_BASIS_CONTENT
    else:
        signature_basis = ROW_SOURCE_SIGNATURE_BASIS_REVISION

    context = _normalize_json(compatibility_context or {})
    spec_payload = {
        "schema_id": ROW_SOURCE_SIGNATURE_SCHEMA_ID,
        "schema_version": ROW_SOURCE_SIGNATURE_SCHEMA_VERSION,
        "algorithm": ROW_SOURCE_SIGNATURE_ALGORITHM,
        "canonicalization": ROW_SOURCE_SIGNATURE_CANONICALIZATION,
        "stage": stage_name,
        "basis": signature_basis,
        "components": [item[0].to_dict() for item in normalized_inputs],
        "compatibility_context": context,
    }
    canonical_json = _canonical_json(spec_payload)
    spec_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    spec = RowSourceSignatureSpec(
        stage=stage_name,
        basis=signature_basis,
        components=tuple(item[0] for item in normalized_inputs),
        compatibility_context=context,
        canonical_json=canonical_json,
        spec_digest=spec_digest,
    )

    row_bytes: list[np.ndarray] = []
    for _, values in normalized_inputs:
        trailing_items = int(np.prod(values.shape[1:], dtype=np.int64))
        row_width_bytes = trailing_items * int(values.dtype.itemsize)
        row_bytes.append(
            np.ascontiguousarray(values)
            .view(np.uint8)
            .reshape(row_count, row_width_bytes)
        )
    signatures = np.empty(
        (row_count, ROW_SOURCE_SIGNATURE_WIDTH_BYTES),
        dtype=np.uint8,
    )
    domain = (
        f"{ROW_SOURCE_SIGNATURE_SCHEMA_ID}\x00{ROW_SOURCE_SIGNATURE_SCHEMA_VERSION}\x00"
        f"{spec_digest}\x00"
    ).encode("utf-8")
    key_bytes = canonical_keys.view(np.uint8).reshape(row_count, 8)
    for row_index in range(row_count):
        hasher = hashlib.sha256()
        hasher.update(domain)
        hasher.update(key_bytes[row_index])
        for component_bytes in row_bytes:
            hasher.update(component_bytes[row_index])
        signatures[row_index] = np.frombuffer(hasher.digest(), dtype=np.uint8)

    return RowSourceSignatureBatch(signatures=signatures, spec=spec)


def load_row_source_signature_spec(
    attrs: Mapping[str, Any],
    *,
    prefix: str = "source_row_signature_",
) -> RowSourceSignatureSpec:
    """Load and verify one persisted signature specification from Zarr attrs."""

    def required(name: str) -> Any:
        key = f"{prefix}{name}"
        if key not in attrs:
            raise RowSourceSignatureError(f"Missing row source signature attr {key!r}.")
        return attrs[key]

    header_expected: dict[str, Any] = {
        "schema_id": ROW_SOURCE_SIGNATURE_SCHEMA_ID,
        "schema_version": ROW_SOURCE_SIGNATURE_SCHEMA_VERSION,
        "algorithm": ROW_SOURCE_SIGNATURE_ALGORITHM,
        "width_bytes": ROW_SOURCE_SIGNATURE_WIDTH_BYTES,
        "canonicalization": ROW_SOURCE_SIGNATURE_CANONICALIZATION,
    }
    for name, expected in header_expected.items():
        actual = _normalize_json(required(name), path=f"{prefix}{name}")
        if actual != expected:
            raise RowSourceSignatureError(
                f"Unsupported row source signature {name}: {actual!r}."
            )

    raw_payload = _normalize_json(required("spec"), path=f"{prefix}spec")
    if not isinstance(raw_payload, dict):
        raise RowSourceSignatureError("Row source signature spec must be a mapping.")
    expected_payload_keys = {
        "schema_id",
        "schema_version",
        "algorithm",
        "canonicalization",
        "stage",
        "basis",
        "components",
        "compatibility_context",
    }
    if set(raw_payload) != expected_payload_keys:
        raise RowSourceSignatureError(
            "Row source signature spec fields do not match schema version 1."
        )
    canonical_json = _canonical_json(raw_payload)
    computed_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    stored_digest = str(required("spec_digest")).strip()
    if stored_digest != computed_digest:
        raise RowSourceSignatureError(
            "Stored row source signature spec digest does not match its specification."
        )

    for name, expected in header_expected.items():
        if name == "width_bytes":
            continue
        if raw_payload.get(name) != expected:
            raise RowSourceSignatureError(
                f"Row source signature spec has incompatible {name}."
            )

    stage = str(raw_payload.get("stage", "")).strip()
    if not stage or str(required("stage")).strip() != stage:
        raise RowSourceSignatureError(
            "Row source signature stage metadata is inconsistent."
        )
    basis = str(raw_payload.get("basis", "")).strip()
    if str(required("basis")).strip() != basis:
        raise RowSourceSignatureError(
            "Row source signature basis metadata is inconsistent."
        )

    raw_components = raw_payload.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise RowSourceSignatureError(
            "Row source signature spec requires at least one component."
        )
    components: list[RowSourceComponentSpec] = []
    seen_names: set[str] = set()
    for index, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, dict):
            raise RowSourceSignatureError(
                f"Row source signature component {index} must be a mapping."
            )
        name = _component_name(raw_component.get("name", ""))
        if name in seen_names:
            raise RowSourceSignatureError(
                f"Row source signature component {name!r} is duplicated."
            )
        seen_names.add(name)
        component_basis = str(raw_component.get("basis", "")).strip()
        if component_basis not in (
            ROW_SOURCE_SIGNATURE_BASIS_CONTENT,
            ROW_SOURCE_SIGNATURE_BASIS_REVISION,
        ):
            raise RowSourceSignatureError(
                f"Row source signature component {name!r} has invalid basis."
            )
        expected_component_keys = {
            "name",
            "basis",
            "dtype",
            "trailing_shape",
        }
        if component_basis == ROW_SOURCE_SIGNATURE_BASIS_REVISION:
            expected_component_keys.add("revision_authority")
        if set(raw_component) != expected_component_keys:
            raise RowSourceSignatureError(
                f"Row source signature component {name!r} fields do not match its basis."
            )
        dtype_text = str(raw_component.get("dtype", "")).strip()
        try:
            dtype = np.dtype(dtype_text)
        except TypeError as exc:
            raise RowSourceSignatureError(
                f"Row source signature component {name!r} has invalid dtype."
            ) from exc
        if component_basis == ROW_SOURCE_SIGNATURE_BASIS_REVISION:
            if dtype.str != "<u8":
                raise RowSourceSignatureError(
                    f"Revision component {name!r} must use canonical <u8 dtype."
                )
            authority = str(raw_component.get("revision_authority", "")).strip()
            if not authority:
                raise RowSourceSignatureError(
                    f"Revision component {name!r} is missing its authority."
                )
        else:
            authority = None
            if "revision_authority" in raw_component:
                raise RowSourceSignatureError(
                    f"Content component {name!r} cannot declare a revision authority."
                )
            if dtype.kind not in "biuf" or int(dtype.itemsize) not in (1, 2, 4, 8):
                raise RowSourceSignatureError(
                    f"Content component {name!r} has an unsupported canonical dtype."
                )
            canonical_dtype_text = (
                "|b1" if dtype.kind == "b" else dtype.newbyteorder("<").str
            )
            if dtype_text != canonical_dtype_text:
                raise RowSourceSignatureError(
                    f"Content component {name!r} dtype is not canonical little-endian."
                )
        raw_shape = raw_component.get("trailing_shape")
        if not isinstance(raw_shape, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in raw_shape
        ):
            raise RowSourceSignatureError(
                f"Row source signature component {name!r} has invalid trailing_shape."
            )
        components.append(
            RowSourceComponentSpec(
                name=name,
                basis=component_basis,
                dtype=dtype_text,
                trailing_shape=tuple(raw_shape),
                revision_authority=authority,
            )
        )

    if components != sorted(components, key=lambda item: (item.name, item.basis)):
        raise RowSourceSignatureError(
            "Row source signature components are not in canonical order."
        )
    component_bases = {component.basis for component in components}
    expected_basis = (
        ROW_SOURCE_SIGNATURE_BASIS_CONTENT_AND_REVISION
        if len(component_bases) == 2
        else next(iter(component_bases))
    )
    if basis != expected_basis:
        raise RowSourceSignatureError(
            "Row source signature aggregate basis does not match its components."
        )
    context = raw_payload.get("compatibility_context")
    if not isinstance(context, dict):
        raise RowSourceSignatureError(
            "Row source signature compatibility_context must be a mapping."
        )
    return RowSourceSignatureSpec(
        stage=stage,
        basis=basis,
        components=tuple(components),
        compatibility_context=context,
        canonical_json=canonical_json,
        spec_digest=computed_digest,
    )


def validate_row_source_signature_array(
    array: Any,
    *,
    expected_row_count: int | None = None,
) -> int:
    """Validate signature array metadata without loading its row payload."""

    shape = tuple(int(value) for value in getattr(array, "shape", ()))
    dtype = np.dtype(getattr(array, "dtype", None))
    if len(shape) != 2 or shape[1] != ROW_SOURCE_SIGNATURE_WIDTH_BYTES:
        raise RowSourceSignatureError("source_row_signature must have shape (N, 32).")
    if dtype != np.dtype(np.uint8):
        raise RowSourceSignatureError("source_row_signature must use uint8 dtype.")
    if expected_row_count is not None and shape[0] != int(expected_row_count):
        raise RowSourceSignatureError(
            "source_row_signature row count does not match its row-aligned source."
        )
    return shape[0]


def assert_row_source_signature_specs_match(
    expected: RowSourceSignatureSpec,
    actual: RowSourceSignatureSpec,
) -> None:
    """Fail closed before comparing signatures built under different specs."""

    if expected.spec_digest != actual.spec_digest:
        raise RowSourceSignatureError(
            "Row source signature specifications differ; row reuse is not safe."
        )


__all__ = [
    "ROW_SOURCE_SIGNATURE_ARRAY",
    "ROW_SOURCE_SIGNATURE_SCHEMA_ID",
    "ROW_SOURCE_SIGNATURE_SCHEMA_VERSION",
    "ROW_SOURCE_SIGNATURE_ALGORITHM",
    "ROW_SOURCE_SIGNATURE_WIDTH_BYTES",
    "ROW_SOURCE_SIGNATURE_CANONICALIZATION",
    "ROW_SOURCE_SIGNATURE_BASIS_CONTENT",
    "ROW_SOURCE_SIGNATURE_BASIS_REVISION",
    "ROW_SOURCE_SIGNATURE_BASIS_CONTENT_AND_REVISION",
    "RowSourceSignatureError",
    "RowSourceComponentSpec",
    "RowSourceSignatureSpec",
    "RowSourceSignatureBatch",
    "build_row_source_signatures",
    "load_row_source_signature_spec",
    "validate_row_source_signature_array",
    "assert_row_source_signature_specs_match",
]
