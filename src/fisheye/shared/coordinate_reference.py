"""Verified reference-extent authorities for canonical coordinate writers.

The compact coordinate descriptor stores only a record reference, digest, and
selector.  Those strings are not writer evidence.  This module derives a
sealed authority from the exact persisted array or attrs node whose values
define the coordinate extent.  Canonical writer APIs can therefore require a
``BoundReferenceExtent`` rather than accepting caller-invented provenance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    archive_identity,
)


ARRAY_REFERENCE_EXTENT_SCHEMA_ID = "palette.array_reference_extent"
ATTRS_REFERENCE_EXTENT_SCHEMA_ID = "palette.attrs_reference_extent"
PERSISTED_RECORD_REFERENCE_EXTENT_SCHEMA_ID = (
    "palette.persisted_record_reference_extent"
)
REFERENCE_EXTENT_SCHEMA_VERSION = 1
REFERENCE_EXTENT_CANONICALIZATION = "canonical_json_sort_keys_v1"
REFERENCE_EXTENT_UNITS = frozenset({"px", "mm"})

_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.:+-]+$")
_ATTR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VERIFIED_REFERENCE_EXTENT_SEAL = object()
_PERSISTED_BINDING_FIELDS = frozenset(
    {
        "bound_record_attr",
        "bound_digest_attr",
        "bound_width_field",
        "bound_height_field",
        "bound_units_field",
    }
)


class CoordinateReferenceError(ValueError):
    """Raised when an extent authority cannot be proved from an exact node."""


def canonical_node_path(node: Any) -> str:
    """Return one canonical archive-relative node path or fail closed."""

    value = getattr(node, "path", None)
    if not isinstance(value, str) or not value:
        raise CoordinateReferenceError(
            "A persisted node must expose a non-empty archive-relative path."
        )
    if (
        value != value.strip()
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
    ):
        raise CoordinateReferenceError(
            f"Persisted node path {value!r} is not canonical archive-relative form."
        )
    segments = value.split("/")
    if any(
        part in {"", ".", ".."} or _PATH_SEGMENT_RE.fullmatch(part) is None
        for part in segments
    ):
        raise CoordinateReferenceError(
            f"Persisted node path {value!r} contains a noncanonical segment."
        )
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    _require_exact_builtin_json(value, path="$")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _require_exact_builtin_json(value: Any, *, path: str) -> None:
    """Reject non-JSON containers and scalar subclasses before digesting.

    A digest over an ``OrderedDict``, NumPy scalar, tuple, or custom mapping can
    serialize to the same bytes as canonical JSON while retaining different
    in-memory semantics.  Persisted coordinate authorities therefore accept
    only exact built-in ``dict``/``list`` containers and JSON scalar types.
    """

    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise CoordinateReferenceError(
                    f"{path} contains a non-string JSON object key."
                )
            _require_exact_builtin_json(item, path=f"{path}.{key}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_exact_builtin_json(item, path=f"{path}[{index}]")
        return
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CoordinateReferenceError(
                f"{path} contains a non-finite JSON number."
            )
        return
    raise CoordinateReferenceError(
        f"{path} must use exact built-in JSON containers and scalar types."
    )


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _positive_dimension(value: Any, *, field_name: str, integral: bool) -> int | float:
    if integral:
        if type(value) is not int or value <= 0:
            raise CoordinateReferenceError(
                f"{field_name} must be an exact positive integer of built-in Python type for pixel units."
            )
        return value
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise CoordinateReferenceError(f"{field_name} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise CoordinateReferenceError(
            f"{field_name} must be positive and finite."
        )
    return number


@dataclass(frozen=True)
class BoundReferenceExtent:
    """Exact width/height evidence, not standalone coordinate semantics.

    Array shapes and generic attrs prove only dimensions.  They become usable
    coordinate evidence only when a typed frame/endpoint authority binds them
    together with space, origin, axes, units, and pixel convention.
    """

    record_ref: str
    record_sha256: str
    selector: str
    width: int | float
    height: int | float
    units: str
    authority_kind: str
    authority_record: Mapping[str, Any]
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _authority_node: Any = field(repr=False, compare=False)
    _verification_seal: object = field(repr=False, compare=False)

    def assert_verified(self) -> None:
        if (
            type(self) is not BoundReferenceExtent
            or self._verification_seal is not _VERIFIED_REFERENCE_EXTENT_SEAL
        ):
            raise CoordinateReferenceError(
                "Reference extent was not derived by a verified authority loader."
            )
        try:
            current_archive = archive_identity(self._authority_node)
        except ArchiveIdentityError as exc:
            raise CoordinateReferenceError(str(exc)) from exc
        if current_archive != self._archive_identity:
            raise CoordinateReferenceError(
                "Reference-extent authority moved to a different archive/store."
            )
        digest_record = self.authority_record
        if self.authority_kind == PERSISTED_RECORD_REFERENCE_EXTENT_SCHEMA_ID:
            digest_record = {
                name: item
                for name, item in self.authority_record.items()
                if name not in _PERSISTED_BINDING_FIELDS
            }
        if _mapping_sha256(digest_record) != self.record_sha256:
            raise CoordinateReferenceError(
                "Reference-extent authority content no longer matches its digest."
            )
        if self.authority_kind == ARRAY_REFERENCE_EXTENT_SCHEMA_ID:
            current = bind_array_reference_extent(
                self._authority_node,
                units=self.units,
            )
        elif self.authority_kind == ATTRS_REFERENCE_EXTENT_SCHEMA_ID:
            match = re.fullmatch(
                r"attrs\[([a-z][a-z0-9_]*),([a-z][a-z0-9_]*)\]",
                self.selector,
            )
            if match is None:
                raise CoordinateReferenceError(
                    "Bound attrs selector is no longer canonical."
                )
            current = bind_attrs_reference_extent(
                self._authority_node,
                width_attr=match.group(1),
                height_attr=match.group(2),
                units=self.units,
            )
        elif self.authority_kind == PERSISTED_RECORD_REFERENCE_EXTENT_SCHEMA_ID:
            record_attr = self.authority_record.get("bound_record_attr")
            digest_attr = self.authority_record.get("bound_digest_attr")
            width_field = self.authority_record.get("bound_width_field")
            height_field = self.authority_record.get("bound_height_field")
            units_field = self.authority_record.get("bound_units_field")
            if not all(
                isinstance(item, str)
                for item in (
                    record_attr,
                    digest_attr,
                    width_field,
                    height_field,
                    units_field,
                )
            ):
                raise CoordinateReferenceError(
                    "Persisted reference-record binding fields are invalid."
                )
            current = bind_persisted_record_reference_extent(
                self._authority_node,
                record_attr=record_attr,
                digest_attr=digest_attr,
                width_field=width_field,
                height_field=height_field,
                units_field=units_field,
            )
        else:
            raise CoordinateReferenceError(
                f"Unsupported bound authority kind {self.authority_kind!r}."
            )
        if (
            current.record_ref != self.record_ref
            or current.record_sha256 != self.record_sha256
            or current.selector != self.selector
            or current.width != self.width
            or current.height != self.height
            or current.units != self.units
            or current.authority_kind != self.authority_kind
            or current.authority_record != self.authority_record
            or current.archive_identity != self.archive_identity
        ):
            raise CoordinateReferenceError(
                "Reference-extent authority changed after it was bound."
            )

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity

    @property
    def authority_scope(self) -> str:
        """Make the deliberately limited authority explicit to consumers."""

        return "extent_only"


def bind_array_reference_extent(
    node: Any,
    *,
    units: str,
) -> BoundReferenceExtent:
    """Bind dimensions only; this does not declare the array's coordinate space."""

    if units not in REFERENCE_EXTENT_UNITS:
        raise CoordinateReferenceError(f"Unsupported reference units {units!r}.")
    path = canonical_node_path(node)
    raw_shape = getattr(node, "shape", None)
    if not isinstance(raw_shape, (tuple, list)) or len(raw_shape) < 2:
        raise CoordinateReferenceError(
            "Array extent authority must expose shape with at least two dimensions."
        )
    shape: list[int] = []
    for index, item in enumerate(raw_shape):
        if type(item) is not int:
            raise CoordinateReferenceError(
                f"Array shape[{index}] must be an exact Python integer."
            )
        dimension = item
        if dimension < 0:
            raise CoordinateReferenceError(
                f"Array shape[{index}] must be nonnegative."
            )
        shape.append(dimension)
    height = _positive_dimension(shape[-2], field_name="height", integral=units == "px")
    width = _positive_dimension(shape[-1], field_name="width", integral=units == "px")
    try:
        dtype = np.dtype(getattr(node, "dtype")).str
    except (AttributeError, TypeError) as exc:
        raise CoordinateReferenceError(
            "Array extent authority must expose a canonical dtype."
        ) from exc
    record = {
        "schema_id": ARRAY_REFERENCE_EXTENT_SCHEMA_ID,
        "schema_version": REFERENCE_EXTENT_SCHEMA_VERSION,
        "array_path": f"/{path}",
        "shape": shape,
        "dtype": dtype,
        "selector": "shape[-2:]",
        "width": width,
        "height": height,
        "units": units,
        "canonicalization": REFERENCE_EXTENT_CANONICALIZATION,
    }
    digest = _mapping_sha256(record)
    return BoundReferenceExtent(
        record_ref=f"/{path}@zarr_metadata",
        record_sha256=digest,
        selector="shape[-2:]",
        width=width,
        height=height,
        units=units,
        authority_kind=ARRAY_REFERENCE_EXTENT_SCHEMA_ID,
        authority_record=record,
        _archive_identity=archive_identity(node),
        _authority_node=node,
        _verification_seal=_VERIFIED_REFERENCE_EXTENT_SEAL,
    )


def bind_attrs_reference_extent(
    node: Any,
    *,
    width_attr: str,
    height_attr: str,
    units: str,
) -> BoundReferenceExtent:
    """Bind dimensions only; caller-supplied units are not coordinate authority."""

    if units not in REFERENCE_EXTENT_UNITS:
        raise CoordinateReferenceError(f"Unsupported reference units {units!r}.")
    for name, label in ((width_attr, "width_attr"), (height_attr, "height_attr")):
        if not isinstance(name, str) or _ATTR_NAME_RE.fullmatch(name) is None:
            raise CoordinateReferenceError(f"{label} is not a canonical attr name.")
    if width_attr == height_attr:
        raise CoordinateReferenceError("Width and height attrs must be distinct.")
    path = canonical_node_path(node)
    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise CoordinateReferenceError("Extent authority node must expose attrs.")
    missing = [name for name in (width_attr, height_attr) if name not in attrs]
    if missing:
        raise CoordinateReferenceError(
            f"Extent authority is missing attrs {missing!r}."
        )
    integral = units == "px"
    width = _positive_dimension(
        attrs[width_attr], field_name=width_attr, integral=integral
    )
    height = _positive_dimension(
        attrs[height_attr], field_name=height_attr, integral=integral
    )
    selector = f"attrs[{width_attr},{height_attr}]"
    record = {
        "schema_id": ATTRS_REFERENCE_EXTENT_SCHEMA_ID,
        "schema_version": REFERENCE_EXTENT_SCHEMA_VERSION,
        "node_path": f"/{path}",
        "selector": selector,
        "width": width,
        "height": height,
        "units": units,
        "canonicalization": REFERENCE_EXTENT_CANONICALIZATION,
    }
    digest = _mapping_sha256(record)
    return BoundReferenceExtent(
        record_ref=f"/{path}",
        record_sha256=digest,
        selector=selector,
        width=width,
        height=height,
        units=units,
        authority_kind=ATTRS_REFERENCE_EXTENT_SCHEMA_ID,
        authority_record=record,
        _archive_identity=archive_identity(node),
        _authority_node=node,
        _verification_seal=_VERIFIED_REFERENCE_EXTENT_SEAL,
    )


def bind_persisted_record_reference_extent(
    node: Any,
    *,
    record_attr: str,
    digest_attr: str,
    width_field: str,
    height_field: str,
    units_field: str,
) -> BoundReferenceExtent:
    """Bind an extent to an exact persisted mapping plus its stored digest.

    The selected width/height must agree both with the mapping and with the
    same-named direct node attrs.  This prevents a valid record digest from
    masking stale or conflicting convenience attrs.
    """

    for name, label in (
        (record_attr, "record_attr"),
        (digest_attr, "digest_attr"),
        (width_field, "width_field"),
        (height_field, "height_field"),
        (units_field, "units_field"),
    ):
        if not isinstance(name, str) or _ATTR_NAME_RE.fullmatch(name) is None:
            raise CoordinateReferenceError(f"{label} is not a canonical name.")
    if units_field != "units":
        raise CoordinateReferenceError(
            "Persisted extent records must use the canonical digested 'units' field; "
            "caller-selected unit aliases could reinterpret one record/digest."
        )
    if width_field == height_field:
        raise CoordinateReferenceError("Width and height fields must be distinct.")
    path = canonical_node_path(node)
    attrs = getattr(node, "attrs", None)
    if not isinstance(attrs, Mapping):
        raise CoordinateReferenceError("Reference-record node must expose attrs.")
    missing = [
        name
        for name in (record_attr, digest_attr, width_field, height_field)
        if name not in attrs
    ]
    if missing:
        raise CoordinateReferenceError(
            f"Reference-record authority is missing attrs {missing!r}."
        )
    raw_record = attrs[record_attr]
    if type(raw_record) is not dict:
        raise CoordinateReferenceError(
            f"Persisted {record_attr!r} must be an exact built-in dict."
        )
    _require_exact_builtin_json(raw_record, path=f"@{record_attr}")
    record = copy.deepcopy(raw_record)
    reserved = sorted(_PERSISTED_BINDING_FIELDS.intersection(record))
    if reserved:
        raise CoordinateReferenceError(
            "Persisted reference record uses reserved binding fields "
            f"{reserved!r}."
        )
    stored_digest = attrs[digest_attr]
    actual_digest = _mapping_sha256(record)
    if not isinstance(stored_digest, str) or stored_digest != actual_digest:
        raise CoordinateReferenceError(
            "Persisted reference-record digest does not match canonical content."
        )
    for field_name in (width_field, height_field):
        if field_name not in record:
            raise CoordinateReferenceError(
                f"Persisted reference record lacks {field_name!r}."
            )
        direct = attrs[field_name]
        recorded = record[field_name]
        if type(direct) is not type(recorded) or direct != recorded:
            raise CoordinateReferenceError(
                f"Direct attr {field_name!r} conflicts with its persisted record."
            )
    if units_field not in record:
        raise CoordinateReferenceError(
            f"Persisted reference record lacks {units_field!r}."
        )
    units = record[units_field]
    if units not in REFERENCE_EXTENT_UNITS:
        raise CoordinateReferenceError(
            f"Persisted reference record has unsupported units {units!r}."
        )
    integral = units == "px"
    width = _positive_dimension(
        record[width_field],
        field_name=width_field,
        integral=integral,
    )
    height = _positive_dimension(
        record[height_field],
        field_name=height_field,
        integral=integral,
    )
    selector = f"attrs[{width_field},{height_field}]"
    binding_record = dict(record)
    binding_record.update(
        {
            "bound_record_attr": record_attr,
            "bound_digest_attr": digest_attr,
            "bound_width_field": width_field,
            "bound_height_field": height_field,
            "bound_units_field": units_field,
        }
    )
    return BoundReferenceExtent(
        record_ref=f"/{path}@{record_attr}",
        record_sha256=actual_digest,
        selector=selector,
        width=width,
        height=height,
        units=units,
        authority_kind=PERSISTED_RECORD_REFERENCE_EXTENT_SCHEMA_ID,
        authority_record=binding_record,
        _archive_identity=archive_identity(node),
        _authority_node=node,
        _verification_seal=_VERIFIED_REFERENCE_EXTENT_SEAL,
    )


def verify_bound_reference_extent(
    value: BoundReferenceExtent,
) -> BoundReferenceExtent:
    if (
        type(value) is not BoundReferenceExtent
        or getattr(value, "_verification_seal", None)
        is not _VERIFIED_REFERENCE_EXTENT_SEAL
    ):
        raise CoordinateReferenceError(
            "Canonical writers require a BoundReferenceExtent."
        )
    value.assert_verified()
    return value


__all__ = [
    "ARRAY_REFERENCE_EXTENT_SCHEMA_ID",
    "ATTRS_REFERENCE_EXTENT_SCHEMA_ID",
    "PERSISTED_RECORD_REFERENCE_EXTENT_SCHEMA_ID",
    "REFERENCE_EXTENT_CANONICALIZATION",
    "REFERENCE_EXTENT_SCHEMA_VERSION",
    "REFERENCE_EXTENT_UNITS",
    "BoundReferenceExtent",
    "CoordinateReferenceError",
    "bind_array_reference_extent",
    "bind_attrs_reference_extent",
    "bind_persisted_record_reference_extent",
    "canonical_node_path",
    "verify_bound_reference_extent",
]
