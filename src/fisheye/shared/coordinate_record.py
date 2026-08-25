"""Sealed bindings for arbitrary persisted coordinate-lineage records.

Canonical coordinate descriptors deliberately stay compact: detailed source,
crop, import, and calibration evidence lives in external persisted records and
the descriptor stores only the record path plus its content digest.  A caller-
constructed path/digest pair is not proof that the record exists.  This module
is the generic exact-node boundary used for lineage records that are not an
extent, row-identity, frame, or directed-transform authority.

The record is one JSON mapping stored in a node attribute with a sibling
``<attr>_sha256`` attribute.  Bindings retain the exact node and re-read both
attributes whenever they are consumed, so stale or replaced evidence fails
closed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Mapping

from fisheye.shared.archive_identity import (
    ArchiveIdentity,
    ArchiveIdentityError,
    archive_identity,
)
from fisheye.shared.coordinate_reference import (
    CoordinateReferenceError,
    canonical_node_path,
)


_ATTR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BOUND_COORDINATE_RECORD_SEAL = object()


class CoordinateRecordError(ValueError):
    """Raised when an external coordinate evidence record is not exact."""


def _canonical_json_value(value: Any, *, path: str) -> Any:
    """Validate exact raw JSON containers/scalars without normalizing them."""

    if type(value) is dict:
        result: dict[str, Any] = {}
        for name, item in value.items():
            if type(name) is not str:
                raise CoordinateRecordError(
                    f"Coordinate record key at {path} must be an exact string."
                )
            result[name] = _canonical_json_value(item, path=f"{path}.{name}")
        return result
    if type(value) is list:
        return [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise CoordinateRecordError(
        f"Coordinate record value at {path} must use exact finite JSON container/scalar types (finite canonical JSON with built-in types)."
    )


def _canonical_record(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise CoordinateRecordError(
            "Coordinate record must be an exact built-in dict."
        )
    if not value:
        raise CoordinateRecordError("Coordinate record must not be empty.")
    result = _canonical_json_value(value, path="$")
    assert type(result) is dict
    return result


def _raw_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(right) is dict:
        return set(left) == set(right) and all(
            _raw_equal(left[name], right[name]) for name in right
        )
    if type(right) is list:
        return len(left) == len(right) and all(
            _raw_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(right) is float and math.isnan(right):
        return math.isnan(left)
    try:
        result = left == right
    except Exception:
        return False
    return bool(result) if type(result) in {bool} else False


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CoordinateRecordError(
            f"Coordinate record must be finite canonical JSON: {exc}."
        ) from exc


def coordinate_record_sha256(value: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 digest of one JSON record."""

    canonical = _canonical_record(value)
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _attr_names(
    attr_name: str,
    digest_attr_name: str | None,
) -> tuple[str, str]:
    if not isinstance(attr_name, str) or _ATTR_NAME_RE.fullmatch(attr_name) is None:
        raise CoordinateRecordError(
            "Coordinate record attr name must be canonical snake_case."
        )
    expected_digest_name = f"{attr_name}_sha256"
    if digest_attr_name is None:
        digest_attr_name = expected_digest_name
    if digest_attr_name != expected_digest_name:
        raise CoordinateRecordError(
            f"Coordinate record digest attr must be {expected_digest_name!r}."
        )
    return attr_name, digest_attr_name


def _attrs(node: Any) -> Any:
    attrs = getattr(node, "attrs", None)
    if attrs is None or not hasattr(attrs, "keys"):
        raise CoordinateRecordError("Persisted coordinate record node has no attrs.")
    return attrs


def _trusted_mutable_attrs(node: Any) -> Any:
    """Return the exact attrs transaction boundary or fail before mutation."""

    attrs = _attrs(node)
    trusted = type(attrs) is dict
    if not trusted:
        try:
            from zarr.core.attributes import Attributes as ZarrAttributes
        except ImportError:  # pragma: no cover - Palette depends on zarr
            ZarrAttributes = None  # type: ignore[assignment,misc]
        trusted = ZarrAttributes is not None and type(attrs) is ZarrAttributes
    if not trusted:
        raise CoordinateRecordError(
            "Persisted coordinate record attrs must be an exact built-in dict "
            "or exact Zarr Attributes implementation; no write was attempted."
        )
    if not all(
        callable(getattr(attrs, name, None))
        for name in ("update", "__setitem__", "__delitem__")
    ):
        raise CoordinateRecordError(
            "Persisted coordinate record attrs are not a trusted mutable "
            "transaction boundary; no write was attempted."
        )
    return attrs


def _restore_attrs(attrs: Any, snapshot: Mapping[str, Any]) -> None:
    for name in tuple(attrs.keys()):
        del attrs[name]
    for name, value in snapshot.items():
        attrs[name] = copy.deepcopy(value)
    if not _raw_equal(dict(attrs), dict(snapshot)):
        raise RuntimeError("restored attrs differ from snapshot")


def _path(node: Any) -> str:
    try:
        return canonical_node_path(node)
    except CoordinateReferenceError as exc:
        raise CoordinateRecordError(str(exc)) from exc


def _stored_record(
    node: Any,
    *,
    attr_name: str,
    digest_attr_name: str,
) -> tuple[dict[str, Any], str]:
    attrs = _attrs(node)
    if attr_name not in attrs:
        raise CoordinateRecordError(
            f"Persisted coordinate record attr {attr_name!r} is missing."
        )
    raw_record = attrs[attr_name]
    try:
        record = _canonical_record(raw_record)
    except CoordinateRecordError as exc:
        raise CoordinateRecordError(
            f"Persisted coordinate record attr {attr_name!r} is noncanonical: {exc}"
        ) from exc
    digest = coordinate_record_sha256(record)
    stored_digest = attrs.get(digest_attr_name)
    if (
        not isinstance(stored_digest, str)
        or _SHA256_RE.fullmatch(stored_digest) is None
        or stored_digest != digest
    ):
        raise CoordinateRecordError(
            f"Persisted coordinate record digest {digest_attr_name!r} is missing, "
            "malformed, or stale."
        )
    return record, digest


@dataclass(frozen=True, init=False)
class BoundCoordinateRecord:
    """Exact persisted external record accepted by canonical publication."""

    record_ref: str
    record_sha256: str
    attr_name: str
    digest_attr_name: str | None
    _archive_identity: ArchiveIdentity = field(repr=False, compare=False)
    _node: Any = field(repr=False, compare=False)
    _record: Mapping[str, Any] = field(repr=False, compare=False)
    _embedded_payload_keys: tuple[str, str] | None = field(
        repr=False,
        compare=False,
    )
    _verification_seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        record_ref: str,
        record_sha256: str,
        attr_name: str,
        digest_attr_name: str | None,
        archive: ArchiveIdentity,
        node: Any,
        record: Mapping[str, Any],
        embedded_payload_keys: tuple[str, str] | None = None,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _BOUND_COORDINATE_RECORD_SEAL:
            raise CoordinateRecordError(
                "Bound coordinate records must be loaded from an exact persisted node."
            )
        object.__setattr__(self, "record_ref", record_ref)
        object.__setattr__(self, "record_sha256", record_sha256)
        object.__setattr__(self, "attr_name", attr_name)
        object.__setattr__(self, "digest_attr_name", digest_attr_name)
        object.__setattr__(self, "_archive_identity", archive)
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_record", copy.deepcopy(dict(record)))
        object.__setattr__(self, "_embedded_payload_keys", embedded_payload_keys)
        object.__setattr__(self, "_verification_seal", _seal)

    @property
    def record(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self._record))

    def assert_verified(self) -> None:
        verify_bound_coordinate_record(self)

    @property
    def archive_identity(self) -> ArchiveIdentity:
        return self._archive_identity


def bind_persisted_coordinate_record(
    node: Any,
    *,
    attr_name: str,
    digest_attr_name: str | None = None,
) -> BoundCoordinateRecord:
    """Load and seal one exact mapping attribute and its sibling digest."""

    attr_name, digest_attr_name = _attr_names(attr_name, digest_attr_name)
    record, digest = _stored_record(
        node,
        attr_name=attr_name,
        digest_attr_name=digest_attr_name,
    )
    return BoundCoordinateRecord(
        record_ref=f"/{_path(node)}@{attr_name}",
        record_sha256=digest,
        attr_name=attr_name,
        digest_attr_name=digest_attr_name,
        archive=archive_identity(node),
        node=node,
        record=record,
        _seal=_BOUND_COORDINATE_RECORD_SEAL,
    )


def _bind_persisted_manifest_coordinate_record(
    node: Any,
    *,
    attr_name: str = "run_manifest",
    payload_key: str = "payload",
    payload_digest_key: str = "payload_digest",
) -> BoundCoordinateRecord:
    """Bind one immutable manifest whose digest is embedded in its envelope.

    The full manifest remains the live persisted record.  ``record_sha256`` is
    the manifest's canonical payload digest, while verification also requires
    the complete envelope to remain byte-for-byte equivalent to the initially
    bound canonical JSON value.  Semantic profile validation remains the
    responsibility of the profile resolver before this generic proof is built.
    """

    if not isinstance(attr_name, str) or _ATTR_NAME_RE.fullmatch(attr_name) is None:
        raise CoordinateRecordError(
            "Manifest coordinate record attr name must be canonical snake_case."
        )
    if not isinstance(payload_key, str) or _ATTR_NAME_RE.fullmatch(payload_key) is None:
        raise CoordinateRecordError("Manifest payload key must be canonical snake_case.")
    if (
        not isinstance(payload_digest_key, str)
        or _ATTR_NAME_RE.fullmatch(payload_digest_key) is None
    ):
        raise CoordinateRecordError(
            "Manifest payload digest key must be canonical snake_case."
        )
    attrs = _attrs(node)
    if attr_name not in attrs:
        raise CoordinateRecordError(
            f"Persisted manifest coordinate record attr {attr_name!r} is missing."
        )
    record = _canonical_record(attrs[attr_name])
    payload = record.get(payload_key)
    if type(payload) is not dict:
        raise CoordinateRecordError("Manifest coordinate payload must be an exact object.")
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    stored_digest = record.get(payload_digest_key)
    if (
        not isinstance(stored_digest, str)
        or _SHA256_RE.fullmatch(stored_digest) is None
        or stored_digest != digest
    ):
        raise CoordinateRecordError(
            "Manifest coordinate payload digest is missing, malformed, or stale."
        )
    return BoundCoordinateRecord(
        record_ref=f"/{_path(node)}@{attr_name}",
        record_sha256=digest,
        attr_name=attr_name,
        digest_attr_name=None,
        archive=archive_identity(node),
        node=node,
        record=record,
        embedded_payload_keys=(payload_key, payload_digest_key),
        _seal=_BOUND_COORDINATE_RECORD_SEAL,
    )


def verify_bound_coordinate_record(value: Any) -> BoundCoordinateRecord:
    """Re-read the exact persisted node and reject forged or stale bindings."""

    if (
        type(value) is not BoundCoordinateRecord
        or getattr(value, "_verification_seal", None)
        is not _BOUND_COORDINATE_RECORD_SEAL
    ):
        raise CoordinateRecordError("A sealed persisted coordinate record is required.")
    if value._embedded_payload_keys is None:
        current = bind_persisted_coordinate_record(
            value._node,
            attr_name=value.attr_name,
            digest_attr_name=value.digest_attr_name,
        )
    else:
        payload_key, payload_digest_key = value._embedded_payload_keys
        current = _bind_persisted_manifest_coordinate_record(
            value._node,
            attr_name=value.attr_name,
            payload_key=payload_key,
            payload_digest_key=payload_digest_key,
        )
    try:
        current_archive = archive_identity(value._node)
    except ArchiveIdentityError as exc:
        raise CoordinateRecordError(str(exc)) from exc
    if (
        current_archive != value._archive_identity
        or current.archive_identity != value.archive_identity
        or current.record_ref != value.record_ref
        or current.record_sha256 != value.record_sha256
        or current.record != value.record
    ):
        raise CoordinateRecordError(
            "Persisted coordinate record changed after it was bound."
        )
    return value


def stamp_and_bind_persisted_coordinate_record(
    node: Any,
    record: Mapping[str, Any],
    *,
    attr_name: str,
    digest_attr_name: str | None = None,
) -> BoundCoordinateRecord:
    """Transactionally stamp one record/digest pair and return its binding."""

    attr_name, digest_attr_name = _attr_names(attr_name, digest_attr_name)
    # Complete record, path, and mutability checks before touching persisted attrs.
    copied = copy.deepcopy(_canonical_record(record))
    digest = coordinate_record_sha256(copied)
    _path(node)
    attrs = _trusted_mutable_attrs(node)
    snapshot = copy.deepcopy(dict(attrs))
    expected = copy.deepcopy(snapshot)
    expected.update(
        {
            attr_name: copy.deepcopy(copied),
            digest_attr_name: digest,
        }
    )
    try:
        attrs.update(
            {
                attr_name: copy.deepcopy(copied),
                digest_attr_name: digest,
            }
        )
        if not _raw_equal(dict(attrs), expected):
            raise CoordinateRecordError(
                "Coordinate record stamp did not preserve the exact full attrs mapping."
            )
        bound = bind_persisted_coordinate_record(
            node,
            attr_name=attr_name,
            digest_attr_name=digest_attr_name,
        )
        if not _raw_equal(dict(attrs), expected):
            raise CoordinateRecordError(
                "Coordinate record attrs changed during post-write verification."
            )
        return bound
    except Exception as exc:
        rollback_error: Exception | None = None
        try:
            _restore_attrs(attrs, snapshot)
        except Exception as rollback_exc:  # pragma: no cover - hostile attrs mapping
            rollback_error = rollback_exc
        if rollback_error is not None:
            raise CoordinateRecordError(
                "Coordinate record stamp failed and rollback was incomplete: "
                f"{rollback_error}."
            ) from exc
        if isinstance(exc, CoordinateRecordError):
            raise
        raise CoordinateRecordError(f"Coordinate record stamp failed: {exc}.") from exc


__all__ = [
    "BoundCoordinateRecord",
    "CoordinateRecordError",
    "bind_persisted_coordinate_record",
    "coordinate_record_sha256",
    "stamp_and_bind_persisted_coordinate_record",
    "verify_bound_coordinate_record",
]
