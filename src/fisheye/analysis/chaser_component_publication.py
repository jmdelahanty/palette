"""Payload-bound publication records for derived chaser components.

Derived groups nested beneath a verified ``chaser_distance`` run are not
automatically authoritative.  This module gives each component an independent,
exact manifest and gives its component-family parent one digest-bound selector.

The functions here deliberately separate three operations:

1. build and persist an immutable component payload;
2. seal and validate that payload without making it selectable; and
3. publish one exact selector envelope after validation succeeds.

Filesystem-level hidden-copy/rename publication is owned by the workflow
publisher.  This module owns the logical record and the fail-closed validation
performed both before and after that rename.
"""

from __future__ import annotations

import hashlib
import json
import math
import copy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.coordinate_frame_record import array_values_sha256


COMPONENT_MANIFEST_ATTR = "chaser_component_publication_manifest"
COMPONENT_MANIFEST_DIGEST_ATTR = f"{COMPONENT_MANIFEST_ATTR}_sha256"
COMPONENT_MANIFEST_SCHEMA_ID = "palette.chaser_component_publication_manifest"
COMPONENT_MANIFEST_SCHEMA_VERSION = 1

COMPONENT_SELECTOR_ATTR = "chaser_component_publication_authority"
COMPONENT_SELECTOR_DIGEST_ATTR = f"{COMPONENT_SELECTOR_ATTR}_sha256"
COMPONENT_SELECTOR_SCHEMA_ID = "palette.chaser_component_publication_authority"
COMPONENT_SELECTOR_SCHEMA_VERSION = 1

COMPONENT_HANDLE_SCHEMA_ID = "palette.chaser_component_dependency_handle"
COMPONENT_HANDLE_SCHEMA_VERSION = 1

_MANIFEST_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "publication_state",
        "selector_eligible",
        "component",
        "base_authority",
        "source_authorities",
        "parameters",
        "payload",
    }
)
_COMPONENT_FIELDS = frozenset(
    {
        "relative_path",
        "component_family",
        "component_name",
        "semantic_schema_id",
        "semantic_schema_version",
        "method_id",
        "method_version",
    }
)
_BASE_FIELDS = frozenset(
    {
        "run_path",
        "publication_seal_ref",
        "publication_seal_sha256",
        "surface_manifest_ref",
        "surface_manifest_sha256",
        "row_identity_ref",
        "row_identity_sha256",
        "read_authority_sha256",
    }
)
_PAYLOAD_FIELDS = frozenset({"groups", "arrays"})
_GROUP_FIELDS = frozenset({"path", "attributes"})
_ARRAY_FIELDS = frozenset({"path", "dtype", "shape", "attributes", "content_sha256"})
_SELECTOR_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "component_family",
        "selected_component",
        "component_path",
        "component_manifest_ref",
        "component_manifest_sha256",
        "base_run_path",
        "base_publication_seal_sha256",
        "approval_state",
    }
)
_HANDLE_BODY_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "resolution_policy",
        "base_run_path",
        "base_publication_seal_sha256",
        "component_family",
        "component_name",
        "component_path",
        "component_manifest_ref",
        "component_manifest_sha256",
        "semantic_schema_id",
        "semantic_schema_version",
        "method_id",
        "method_version",
    }
)
_HANDLE_FIELDS = frozenset({*_HANDLE_BODY_FIELDS, "record_sha256"})

# These component-root attributes are owned by the mechanical publisher.  They
# are explicitly non-scientific and are excluded so hidden-copy ownership,
# completion receipts, and the final eligibility commit cannot invalidate the
# already-sealed scientific payload.  Every other root attribute is protected.
NON_AUTHORITATIVE_PUBLISHER_ATTRS = frozenset(
    {
        "atomic_publication_owner_uuid",
        "cluster_output_staging",
        "palette_run_completed_at_utc",
        "palette_run_completion_status",
        "palette_run_error",
        "publication_status",
        "stage_selector_eligible",
    }
)

# Maintained component builders stamp these operational wall-clock values while
# recreating an otherwise identical node-local candidate.  They remain present
# in, and protected by, each immutable component manifest.  Only the explicit
# retry-equivalence helper below redacts them so a caller can reconstruct a
# lost post-commit acknowledgement without pretending a new timestamp changes
# the scientific payload.  No other attribute is ignored during recovery.
RETRY_TRANSIENT_ATTRIBUTE_NAMES = frozenset(
    {
        "completed_at_utc",
        "created_at_utc",
        "generated_at",
        "generated_at_utc",
    }
)


class ChaserComponentPublicationError(ValueError):
    """Raised when a derived component cannot be sealed or selected exactly."""


def _fail(message: str) -> None:
    raise ChaserComponentPublicationError(message)


@dataclass(frozen=True)
class VerifiedChaserComponent:
    """Detached, read-only payload from one digest-bound selected component."""

    base_run_path: str
    component_family: str
    component_name: str
    component_path: str
    semantic_schema_id: str
    semantic_schema_version: int
    method_id: str
    method_version: str
    manifest_sha256: str
    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    group_attributes: Mapping[str, Mapping[str, Any]]

    def array(self, relative_path: str) -> np.ndarray:
        path = _relative_path(relative_path, label="component array path")
        try:
            return self.arrays[path]
        except KeyError:
            _fail(f"Verified component has no declared array {path!r}.")


@dataclass(frozen=True)
class VerifiedChaserComponentGroup:
    """Read-only group handle validated against one explicit dependency record."""

    base_run_path: str
    component_family: str
    component_name: str
    component_path: str
    semantic_schema_id: str
    semantic_schema_version: int
    method_id: str
    method_version: str
    manifest_sha256: str
    manifest: Mapping[str, Any]
    dependency_handle: Mapping[str, Any]
    group: Any


def _controlled_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} must be one non-empty controlled name.")
    normalized = value.strip()
    if not normalized or normalized in {".", ".."} or "/" in normalized:
        _fail(f"{label} must be one non-empty controlled name.")
    return normalized


def _relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} must be one exact relative path.")
    normalized = "/".join(part for part in value.strip("/").split("/") if part)
    if not normalized or any(part in {".", ".."} for part in normalized.split("/")):
        _fail(f"{label} must be one exact relative path.")
    return normalized


def _schema_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be one non-empty schema identifier.")
    return value.strip()


def _positive_version(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{label} must be one positive integer.")
    return int(value)


def _sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be one lowercase SHA-256 digest.")
    return value


def _require_runtime_publication_state(
    component: Any,
    *,
    require_selector_eligible: bool,
) -> None:
    """Reject incomplete, tombstoned, or improperly eligible components."""

    attrs = component.attrs
    if attrs.get("palette_run_completion_status") != "complete":
        _fail("Chaser component is not runtime-complete.")
    if "atomic_publication_tombstone" in attrs:
        _fail("Chaser component is an immutable failed-publication tombstone.")
    eligibility = attrs.get("stage_selector_eligible")
    if type(eligibility) is not bool:
        _fail("Chaser component lacks a literal runtime eligibility state.")
    if require_selector_eligible and eligibility is not True:
        _fail("Selected chaser component is not selector eligible.")


def _canonical_value(value: Any, *, path: str = "$") -> Any:
    if isinstance(value, np.generic):
        return _canonical_value(value.item(), path=path)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{path} contains a non-finite float.")
        return float(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{path} contains a non-string mapping key.")
            if key in result:
                _fail(f"{path} contains duplicate key {key!r}.")
            result[key] = _canonical_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(f"{path} contains unsupported JSON value {type(value).__name__}.")


def _redact_retry_transient_attributes(value: Any) -> Any:
    """Remove only explicit operational timestamps from an attribute value."""

    if isinstance(value, Mapping):
        return {
            str(key): _redact_retry_transient_attributes(item)
            for key, item in value.items()
            if str(key) not in RETRY_TRANSIENT_ATTRIBUTE_NAMES
        }
    if isinstance(value, list):
        return [_redact_retry_transient_attributes(item) for item in value]
    return copy.deepcopy(value)


def canonical_component_json_bytes(value: Any) -> bytes:
    """Return strict, deterministic UTF-8 JSON for one publication record."""

    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def component_record_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_component_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ChaserComponentContract:
    """Semantic inputs required to mint one exact component manifest."""

    component_family: str
    component_name: str
    semantic_schema_id: str
    semantic_schema_version: int
    method_id: str
    method_version: str
    parameters: Mapping[str, Any]
    source_authorities: Mapping[str, Any]

    def normalized_component(self, *, relative_path: str) -> dict[str, Any]:
        family = _controlled_name(self.component_family, label="component family")
        name = _controlled_name(self.component_name, label="component name")
        path = _relative_path(relative_path, label="component path")
        if path != f"{family}/{name}":
            _fail("Component relative path must equal its exact family/name binding.")
        method_id = _schema_id(self.method_id, label="method id")
        method_version = _schema_id(self.method_version, label="method version")
        return {
            "relative_path": path,
            "component_family": family,
            "component_name": name,
            "semantic_schema_id": _schema_id(
                self.semantic_schema_id,
                label="component semantic schema id",
            ),
            "semantic_schema_version": _positive_version(
                self.semantic_schema_version,
                label="component semantic schema version",
            ),
            "method_id": method_id,
            "method_version": method_version,
        }


def _exact_fields(
    value: Any, expected: frozenset[str], *, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(f"{label} must contain exactly {sorted(expected)!r}.")
    return value


def _base_authority(snapshot: Any) -> dict[str, Any]:
    run_path = _relative_path(snapshot.run_path, label="base run path")
    authority = snapshot.authority_record()
    return {
        "run_path": run_path,
        "publication_seal_ref": str(snapshot.publication_seal_ref),
        "publication_seal_sha256": _sha256(
            snapshot.publication_seal_sha256,
            label="base publication seal digest",
        ),
        "surface_manifest_ref": str(snapshot.surface_manifest_ref),
        "surface_manifest_sha256": _sha256(
            snapshot.surface_manifest_sha256,
            label="base surface manifest digest",
        ),
        "row_identity_ref": str(snapshot.row_identity_ref),
        "row_identity_sha256": _sha256(
            snapshot.row_identity_sha256,
            label="base row identity digest",
        ),
        "read_authority_sha256": component_record_sha256(authority),
    }


def _attrs(node: Any, *, omit_manifest: bool = False) -> dict[str, Any]:
    values = dict(node.attrs)
    if omit_manifest:
        values.pop(COMPONENT_MANIFEST_ATTR, None)
        values.pop(COMPONENT_MANIFEST_DIGEST_ATTR, None)
        for name in NON_AUTHORITATIVE_PUBLISHER_ATTRS:
            values.pop(name, None)
    normalized = _canonical_value(values, path="$.attributes")
    if not isinstance(normalized, dict):
        _fail("Zarr attributes must normalize to one object.")
    return normalized


def _payload_inventory(component: Any) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    arrays: list[dict[str, Any]] = []

    def visit(group: Any, relative_group: str) -> None:
        groups.append(
            {
                "path": relative_group,
                "attributes": _attrs(group, omit_manifest=not relative_group),
            }
        )
        for name in sorted(str(value) for value in group.array_keys()):
            node = group[name]
            dtype = np.dtype(node.dtype)
            if dtype.hasobject:
                _fail(f"Component array {name!r} uses forbidden object dtype.")
            path = f"{relative_group}/{name}" if relative_group else name
            values = np.asarray(node[:])
            shape = [int(value) for value in node.shape]
            if list(values.shape) != shape or np.dtype(values.dtype) != dtype:
                _fail(f"Component array {path!r} changed while being inventoried.")
            arrays.append(
                {
                    "path": path,
                    "dtype": dtype.str,
                    "shape": shape,
                    "attributes": _attrs(node),
                    "content_sha256": array_values_sha256(values),
                }
            )
        for name in sorted(str(value) for value in group.group_keys()):
            path = f"{relative_group}/{name}" if relative_group else name
            visit(group[name], path)

    visit(component, "")
    if not arrays:
        _fail("A sealed chaser component must contain at least one array.")
    return {"groups": groups, "arrays": arrays}


def build_chaser_component_manifest(
    component: Any,
    *,
    snapshot: Any,
    relative_path: str,
    contract: ChaserComponentContract,
) -> dict[str, Any]:
    """Build an exact ineligible manifest from the persisted component payload."""

    return {
        "schema_id": COMPONENT_MANIFEST_SCHEMA_ID,
        "schema_version": COMPONENT_MANIFEST_SCHEMA_VERSION,
        "publication_state": "complete",
        "selector_eligible": False,
        "component": contract.normalized_component(relative_path=relative_path),
        "base_authority": _base_authority(snapshot),
        "source_authorities": _canonical_value(
            contract.source_authorities,
            path="$.source_authorities",
        ),
        "parameters": _canonical_value(contract.parameters, path="$.parameters"),
        "payload": _payload_inventory(component),
    }


def _validate_manifest_shape(manifest: Any) -> Mapping[str, Any]:
    record = _exact_fields(manifest, _MANIFEST_FIELDS, label="component manifest")
    if (
        record["schema_id"] != COMPONENT_MANIFEST_SCHEMA_ID
        or record["schema_version"] != COMPONENT_MANIFEST_SCHEMA_VERSION
        or record["publication_state"] != "complete"
        or record["selector_eligible"] is not False
    ):
        _fail("Component manifest has an unsupported identity or lifecycle state.")
    component = _exact_fields(
        record["component"], _COMPONENT_FIELDS, label="component identity"
    )
    _relative_path(component["relative_path"], label="component relative path")
    _controlled_name(component["component_family"], label="component family")
    _controlled_name(component["component_name"], label="component name")
    _schema_id(component["semantic_schema_id"], label="semantic schema id")
    _positive_version(
        component["semantic_schema_version"], label="semantic schema version"
    )
    _schema_id(component["method_id"], label="method id")
    _schema_id(component["method_version"], label="method version")
    base = _exact_fields(record["base_authority"], _BASE_FIELDS, label="base authority")
    _relative_path(base["run_path"], label="base run path")
    for field in (
        "publication_seal_sha256",
        "surface_manifest_sha256",
        "row_identity_sha256",
        "read_authority_sha256",
    ):
        _sha256(base[field], label=field)
    payload = _exact_fields(record["payload"], _PAYLOAD_FIELDS, label="payload")
    if not isinstance(payload["groups"], list) or not isinstance(
        payload["arrays"], list
    ):
        _fail("Component payload declarations must be ordered lists.")
    for group in payload["groups"]:
        declaration = _exact_fields(group, _GROUP_FIELDS, label="group declaration")
        if not isinstance(declaration["path"], str) or not isinstance(
            declaration["attributes"], Mapping
        ):
            _fail("Component group declaration is malformed.")
    for array in payload["arrays"]:
        declaration = _exact_fields(array, _ARRAY_FIELDS, label="array declaration")
        _relative_path(declaration["path"], label="array path")
        try:
            dtype = np.dtype(declaration["dtype"])
        except (TypeError, ValueError) as exc:
            _fail(f"Component array dtype is invalid: {exc}.")
        if dtype.hasobject:
            _fail("Component manifest declares forbidden object dtype.")
        if not isinstance(declaration["shape"], list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in declaration["shape"]
        ):
            _fail("Component array shape is invalid.")
        if not isinstance(declaration["attributes"], Mapping):
            _fail("Component array attributes are malformed.")
        _sha256(declaration["content_sha256"], label="array content digest")
    canonical_component_json_bytes(record)
    return record


def persist_chaser_component_manifest(
    component: Any,
    *,
    snapshot: Any,
    relative_path: str,
    contract: ChaserComponentContract,
) -> tuple[dict[str, Any], str]:
    """Seal one complete component without changing a selection pointer."""

    if COMPONENT_MANIFEST_ATTR in component.attrs:
        _fail("Refusing to rewrite an existing chaser component manifest.")
    manifest = build_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path=relative_path,
        contract=contract,
    )
    digest = component_record_sha256(manifest)
    component.attrs[COMPONENT_MANIFEST_ATTR] = manifest
    component.attrs[COMPONENT_MANIFEST_DIGEST_ATTR] = digest
    validate_chaser_component_manifest(
        component,
        snapshot=snapshot,
        expected_relative_path=relative_path,
        expected_contract=contract,
    )
    return manifest, digest


def validate_chaser_component_manifest(
    component: Any,
    *,
    snapshot: Any,
    expected_relative_path: str,
    expected_contract: ChaserComponentContract | None = None,
    expected_manifest_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Recompute all declarations and reject stale or semantically changed payloads."""

    manifest = _validate_manifest_shape(component.attrs.get(COMPONENT_MANIFEST_ATTR))
    digest = _sha256(
        component.attrs.get(COMPONENT_MANIFEST_DIGEST_ATTR),
        label="component manifest digest",
    )
    if component_record_sha256(manifest) != digest:
        _fail("Component manifest digest does not match its canonical payload.")
    if expected_manifest_sha256 is not None and digest != _sha256(
        expected_manifest_sha256, label="expected component manifest digest"
    ):
        _fail("Selected component manifest digest changed.")
    expected_path = _relative_path(
        expected_relative_path, label="expected component relative path"
    )
    component_record = manifest["component"]
    if component_record["relative_path"] != expected_path:
        _fail("Component manifest is bound to a different relative path.")
    if manifest["base_authority"] != _base_authority(snapshot):
        _fail("Component manifest is bound to a different base authority.")
    if expected_contract is None:
        expected_contract = ChaserComponentContract(
            component_family=component_record["component_family"],
            component_name=component_record["component_name"],
            semantic_schema_id=component_record["semantic_schema_id"],
            semantic_schema_version=component_record["semantic_schema_version"],
            method_id=component_record["method_id"],
            method_version=component_record["method_version"],
            parameters=manifest["parameters"],
            source_authorities=manifest["source_authorities"],
        )
    expected = build_chaser_component_manifest(
        component,
        snapshot=snapshot,
        relative_path=expected_path,
        contract=expected_contract,
    )
    if canonical_component_json_bytes(manifest) != canonical_component_json_bytes(
        expected
    ):
        _fail("Component payload or semantic contract changed after sealing.")
    return manifest


def chaser_component_retry_equivalence_sha256(manifest: Any) -> str:
    """Digest one manifest while redacting only operational timestamps.

    This identity is narrower than the persisted manifest digest and is valid
    only for reconstructing acknowledgement of an already-committed,
    selector-ineligible component.  Exact manifest validation still protects
    each candidate independently before this comparison is made.
    """

    record = copy.deepcopy(dict(_validate_manifest_shape(manifest)))
    payload = record["payload"]
    for declaration in (*payload["groups"], *payload["arrays"]):
        declaration["attributes"] = _redact_retry_transient_attributes(
            declaration["attributes"]
        )
    return component_record_sha256(record)


def build_chaser_component_selector(
    *,
    snapshot: Any,
    relative_path: str,
    component_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the only selector envelope accepted for a sealed component family."""

    path = _relative_path(relative_path, label="selected component path")
    parts = path.split("/")
    if len(parts) != 2:
        _fail("Selected component path must be exactly family/name.")
    family, name = parts
    base = _base_authority(snapshot)
    return {
        "schema_id": COMPONENT_SELECTOR_SCHEMA_ID,
        "schema_version": COMPONENT_SELECTOR_SCHEMA_VERSION,
        "component_family": family,
        "selected_component": name,
        "component_path": path,
        "component_manifest_ref": (
            f"/{base['run_path']}/{path}@{COMPONENT_MANIFEST_ATTR}"
        ),
        "component_manifest_sha256": _sha256(
            component_manifest_sha256,
            label="component manifest digest",
        ),
        "base_run_path": base["run_path"],
        "base_publication_seal_sha256": base["publication_seal_sha256"],
        "approval_state": "approved",
    }


def persist_chaser_component_selector(
    component_parent: Any,
    *,
    component: Any,
    snapshot: Any,
    relative_path: str,
) -> tuple[dict[str, Any], str]:
    """Validate one sealed payload and atomically replace its logical selector attrs.

    The two attribute writes are not the filesystem commit point.  Callers must
    perform this only after the hidden component directory has been validated and
    renamed into its immutable final path.
    """

    manifest = validate_chaser_component_manifest(
        component,
        snapshot=snapshot,
        expected_relative_path=relative_path,
    )
    manifest_digest = component_record_sha256(manifest)
    selector = build_chaser_component_selector(
        snapshot=snapshot,
        relative_path=relative_path,
        component_manifest_sha256=manifest_digest,
    )
    selector_digest = component_record_sha256(selector)
    component_parent.attrs[COMPONENT_SELECTOR_ATTR] = selector
    component_parent.attrs[COMPONENT_SELECTOR_DIGEST_ATTR] = selector_digest
    validate_chaser_component_selector(
        component_parent,
        component=component,
        snapshot=snapshot,
        expected_family=selector["component_family"],
    )
    return selector, selector_digest


def validate_chaser_component_selector(
    component_parent: Any,
    *,
    component: Any,
    snapshot: Any,
    expected_family: str,
) -> Mapping[str, Any]:
    """Resolve no fallback: validate exactly the digest-bound selected component."""

    selector = _exact_fields(
        component_parent.attrs.get(COMPONENT_SELECTOR_ATTR),
        _SELECTOR_FIELDS,
        label="component selector",
    )
    if (
        selector["schema_id"] != COMPONENT_SELECTOR_SCHEMA_ID
        or selector["schema_version"] != COMPONENT_SELECTOR_SCHEMA_VERSION
        or selector["approval_state"] != "approved"
    ):
        _fail("Component selector has an unsupported identity or approval state.")
    selector_digest = _sha256(
        component_parent.attrs.get(COMPONENT_SELECTOR_DIGEST_ATTR),
        label="component selector digest",
    )
    if component_record_sha256(selector) != selector_digest:
        _fail("Component selector digest does not match its canonical payload.")
    family = _controlled_name(expected_family, label="expected component family")
    if selector["component_family"] != family:
        _fail("Component selector belongs to a different component family.")
    expected_path = f"{family}/{_controlled_name(selector['selected_component'], label='selected component')}"
    if selector["component_path"] != expected_path:
        _fail("Component selector path is not its exact family/name binding.")
    expected = build_chaser_component_selector(
        snapshot=snapshot,
        relative_path=expected_path,
        component_manifest_sha256=selector["component_manifest_sha256"],
    )
    if canonical_component_json_bytes(selector) != canonical_component_json_bytes(
        expected
    ):
        _fail("Component selector changed or belongs to a different base run.")
    validate_chaser_component_manifest(
        component,
        snapshot=snapshot,
        expected_relative_path=expected_path,
        expected_manifest_sha256=selector["component_manifest_sha256"],
    )
    return selector


def build_chaser_component_handle(
    component: Any,
    *,
    snapshot: Any,
    relative_path: str,
) -> dict[str, Any]:
    """Build one portable handle for an exact sealed component dependency.

    Unlike a selector, this record grants no archive-wide authority.  It lets a
    workflow pass one already-validated immutable candidate to a downstream
    component without writing ``latest`` or making the candidate selectable.
    """

    path = _relative_path(relative_path, label="component dependency path")
    _require_runtime_publication_state(
        component,
        require_selector_eligible=False,
    )
    manifest = validate_chaser_component_manifest(
        component,
        snapshot=snapshot,
        expected_relative_path=path,
    )
    identity = manifest["component"]
    base = _base_authority(snapshot)
    full_path = f"{base['run_path']}/{path}"
    body = {
        "schema_id": COMPONENT_HANDLE_SCHEMA_ID,
        "schema_version": COMPONENT_HANDLE_SCHEMA_VERSION,
        "resolution_policy": "explicit_manifest_no_selector_fallback",
        "base_run_path": base["run_path"],
        "base_publication_seal_sha256": base["publication_seal_sha256"],
        "component_family": identity["component_family"],
        "component_name": identity["component_name"],
        "component_path": full_path,
        "component_manifest_ref": f"/{full_path}@{COMPONENT_MANIFEST_ATTR}",
        "component_manifest_sha256": component_record_sha256(manifest),
        "semantic_schema_id": identity["semantic_schema_id"],
        "semantic_schema_version": identity["semantic_schema_version"],
        "method_id": identity["method_id"],
        "method_version": identity["method_version"],
    }
    return {**body, "record_sha256": component_record_sha256(body)}


def validate_chaser_component_handle(
    handle: Any,
    *,
    snapshot: Any,
) -> Mapping[str, Any]:
    """Validate one explicit component handle without consulting a selector."""

    record = _exact_fields(handle, _HANDLE_FIELDS, label="component dependency handle")
    body = {key: record[key] for key in _HANDLE_BODY_FIELDS}
    if (
        body["schema_id"] != COMPONENT_HANDLE_SCHEMA_ID
        or body["schema_version"] != COMPONENT_HANDLE_SCHEMA_VERSION
        or body["resolution_policy"]
        != "explicit_manifest_no_selector_fallback"
    ):
        _fail("Component dependency handle has an unsupported identity or policy.")
    digest = _sha256(record["record_sha256"], label="component handle digest")
    if component_record_sha256(body) != digest:
        _fail("Component dependency handle digest does not match its canonical payload.")

    base = _base_authority(snapshot)
    if (
        body["base_run_path"] != base["run_path"]
        or body["base_publication_seal_sha256"]
        != base["publication_seal_sha256"]
    ):
        _fail("Component dependency handle belongs to a different base authority.")
    family = _controlled_name(body["component_family"], label="component family")
    name = _controlled_name(body["component_name"], label="component name")
    expected_path = f"{base['run_path']}/{family}/{name}"
    if body["component_path"] != expected_path:
        _fail("Component dependency handle path is not its exact base/family/name binding.")
    if body["component_manifest_ref"] != (
        f"/{expected_path}@{COMPONENT_MANIFEST_ATTR}"
    ):
        _fail("Component dependency handle has a different manifest reference.")
    _sha256(
        body["component_manifest_sha256"],
        label="component manifest digest",
    )
    _schema_id(body["semantic_schema_id"], label="semantic schema id")
    _positive_version(
        body["semantic_schema_version"], label="semantic schema version"
    )
    _schema_id(body["method_id"], label="method id")
    _schema_id(body["method_version"], label="method version")
    canonical_component_json_bytes(record)
    return record


def load_chaser_component_handle_json(path: Path | str) -> Mapping[str, Any]:
    """Read one strict JSON handle document without resolving archive state."""

    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _fail(f"Unable to read strict component dependency handle JSON: {exc}.")
    if not isinstance(value, Mapping):
        _fail("Component dependency handle JSON must contain one object.")
    return MappingProxyType(copy.deepcopy(dict(value)))


def _copy_verified_chaser_component(
    component: Any,
    *,
    snapshot: Any,
    relative_path: str,
    expected_manifest_sha256: str,
    expected_semantic_schema_id: str,
    expected_semantic_schema_version: int,
) -> VerifiedChaserComponent:
    manifest = validate_chaser_component_manifest(
        component,
        snapshot=snapshot,
        expected_relative_path=relative_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    identity = manifest["component"]
    expected_schema_id = _schema_id(
        expected_semantic_schema_id,
        label="expected semantic schema id",
    )
    expected_schema_version = _positive_version(
        expected_semantic_schema_version,
        label="expected semantic schema version",
    )
    if (
        identity["semantic_schema_id"] != expected_schema_id
        or identity["semantic_schema_version"] != expected_schema_version
    ):
        _fail("Selected component has an incompatible semantic schema identity.")

    arrays: dict[str, np.ndarray] = {}
    for declaration in manifest["payload"]["arrays"]:
        path = declaration["path"]
        try:
            node = component[path]
            values = np.array(node[:], copy=True, order="C")
        except Exception as exc:
            _fail(f"Unable to copy selected component array {path!r}: {exc}.")
        if (
            list(values.shape) != declaration["shape"]
            or np.dtype(values.dtype).str != declaration["dtype"]
            or array_values_sha256(values) != declaration["content_sha256"]
        ):
            _fail(f"Selected component array {path!r} changed while being copied.")
        values.setflags(write=False)
        arrays[path] = values
    group_attributes = {
        declaration["path"]: MappingProxyType(
            copy.deepcopy(dict(declaration["attributes"]))
        )
        for declaration in manifest["payload"]["groups"]
    }
    family = identity["component_family"]
    name = identity["component_name"]
    return VerifiedChaserComponent(
        base_run_path=snapshot.run_path,
        component_family=family,
        component_name=name,
        component_path=f"{snapshot.run_path}/{family}/{name}",
        semantic_schema_id=identity["semantic_schema_id"],
        semantic_schema_version=identity["semantic_schema_version"],
        method_id=identity["method_id"],
        method_version=identity["method_version"],
        manifest_sha256=expected_manifest_sha256,
        manifest=MappingProxyType(copy.deepcopy(dict(manifest))),
        arrays=MappingProxyType(arrays),
        group_attributes=MappingProxyType(group_attributes),
    )


def load_explicit_chaser_component(
    root: Any,
    *,
    snapshot: Any,
    handle: Any,
    expected_semantic_schema_id: str,
    expected_semantic_schema_version: int,
) -> VerifiedChaserComponent:
    """Load one digest-bound candidate without selector or child fallback."""

    verified = open_explicit_chaser_component_group(
        root,
        snapshot=snapshot,
        handle=handle,
        expected_semantic_schema_id=expected_semantic_schema_id,
        expected_semantic_schema_version=expected_semantic_schema_version,
    )
    relative_path = f"{verified.component_family}/{verified.component_name}"
    return _copy_verified_chaser_component(
        verified.group,
        snapshot=snapshot,
        relative_path=relative_path,
        expected_manifest_sha256=verified.manifest_sha256,
        expected_semantic_schema_id=expected_semantic_schema_id,
        expected_semantic_schema_version=expected_semantic_schema_version,
    )


def open_explicit_chaser_component_group(
    root: Any,
    *,
    snapshot: Any,
    handle: Any,
    expected_semantic_schema_id: str,
    expected_semantic_schema_version: int,
) -> VerifiedChaserComponentGroup:
    """Open one exact group after validating its handle and sealed payload."""

    record = validate_chaser_component_handle(handle, snapshot=snapshot)
    if (
        record["semantic_schema_id"] != expected_semantic_schema_id
        or record["semantic_schema_version"] != expected_semantic_schema_version
    ):
        _fail("Component dependency handle has an incompatible semantic schema.")
    try:
        component = root[record["component_path"]]
    except Exception as exc:
        _fail(
            f"Explicit component {record['component_path']!r} is unavailable: {exc}."
        )
    _require_runtime_publication_state(
        component,
        require_selector_eligible=False,
    )
    relative_path = f"{record['component_family']}/{record['component_name']}"
    manifest = validate_chaser_component_manifest(
        component,
        snapshot=snapshot,
        expected_relative_path=relative_path,
        expected_manifest_sha256=record["component_manifest_sha256"],
    )
    identity = manifest["component"]
    if (
        identity["semantic_schema_id"] != expected_semantic_schema_id
        or identity["semantic_schema_version"] != expected_semantic_schema_version
        or identity["method_id"] != record["method_id"]
        or identity["method_version"] != record["method_version"]
    ):
        _fail("Explicit component semantic identity differs from its handle.")
    return VerifiedChaserComponentGroup(
        base_run_path=snapshot.run_path,
        component_family=record["component_family"],
        component_name=record["component_name"],
        component_path=record["component_path"],
        semantic_schema_id=record["semantic_schema_id"],
        semantic_schema_version=record["semantic_schema_version"],
        method_id=record["method_id"],
        method_version=record["method_version"],
        manifest_sha256=record["component_manifest_sha256"],
        manifest=MappingProxyType(copy.deepcopy(dict(manifest))),
        dependency_handle=MappingProxyType(copy.deepcopy(dict(record))),
        group=component,
    )


def load_selected_chaser_component(
    root: Any,
    *,
    snapshot: Any,
    component_family: str,
    expected_semantic_schema_id: str,
    expected_semantic_schema_version: int,
) -> VerifiedChaserComponent:
    """Load one selected component with no latest, scan, or raw-child fallback."""

    family = _controlled_name(component_family, label="component family")
    parent_path = f"{_relative_path(snapshot.run_path, label='base run path')}/{family}"
    try:
        parent = root[parent_path]
    except Exception as exc:
        _fail(f"Selected component family {parent_path!r} is unavailable: {exc}.")
    raw_selector = _exact_fields(
        parent.attrs.get(COMPONENT_SELECTOR_ATTR),
        _SELECTOR_FIELDS,
        label="component selector",
    )
    if raw_selector.get("component_family") != family:
        _fail("Component selector belongs to a different component family.")
    name = _controlled_name(
        raw_selector.get("selected_component"), label="selected component"
    )
    try:
        component = parent[name]
    except Exception as exc:
        _fail(f"Selected component {family}/{name!s} is unavailable: {exc}.")
    selector = validate_chaser_component_selector(
        parent,
        component=component,
        snapshot=snapshot,
        expected_family=family,
    )
    _require_runtime_publication_state(
        component,
        require_selector_eligible=True,
    )
    return _copy_verified_chaser_component(
        component,
        snapshot=snapshot,
        relative_path=selector["component_path"],
        expected_manifest_sha256=selector["component_manifest_sha256"],
        expected_semantic_schema_id=expected_semantic_schema_id,
        expected_semantic_schema_version=expected_semantic_schema_version,
    )


__all__ = [
    "COMPONENT_MANIFEST_ATTR",
    "COMPONENT_MANIFEST_DIGEST_ATTR",
    "COMPONENT_MANIFEST_SCHEMA_ID",
    "COMPONENT_MANIFEST_SCHEMA_VERSION",
    "COMPONENT_SELECTOR_ATTR",
    "COMPONENT_SELECTOR_DIGEST_ATTR",
    "COMPONENT_SELECTOR_SCHEMA_ID",
    "COMPONENT_SELECTOR_SCHEMA_VERSION",
    "COMPONENT_HANDLE_SCHEMA_ID",
    "COMPONENT_HANDLE_SCHEMA_VERSION",
    "NON_AUTHORITATIVE_PUBLISHER_ATTRS",
    "RETRY_TRANSIENT_ATTRIBUTE_NAMES",
    "ChaserComponentContract",
    "ChaserComponentPublicationError",
    "VerifiedChaserComponent",
    "VerifiedChaserComponentGroup",
    "build_chaser_component_manifest",
    "build_chaser_component_selector",
    "build_chaser_component_handle",
    "canonical_component_json_bytes",
    "chaser_component_retry_equivalence_sha256",
    "component_record_sha256",
    "persist_chaser_component_manifest",
    "persist_chaser_component_selector",
    "load_explicit_chaser_component",
    "load_chaser_component_handle_json",
    "open_explicit_chaser_component_group",
    "load_selected_chaser_component",
    "validate_chaser_component_handle",
    "validate_chaser_component_manifest",
    "validate_chaser_component_selector",
]
