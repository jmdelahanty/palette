"""Immutable authority records for selector-ineligible coordinate successors.

The records in this module do not authorize a production selector.  They bind a
new coordinate-complete run to one already reviewed immutable source authority
and to byte/logical-equivalent observation payloads.  Canary readers may consume
the successor only when both the source authority and the persisted coordinate
records still validate exactly.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from fisheye.shared.coordinate_record import bind_persisted_coordinate_record
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


COORDINATE_SUCCESSOR_AUTHORITY_ATTR = "coordinate_successor_authority"
COORDINATE_SUCCESSOR_AUTHORITY_DIGEST_ATTR = (
    f"{COORDINATE_SUCCESSOR_AUTHORITY_ATTR}_sha256"
)
COORDINATE_SUCCESSOR_AUTHORITY_SCHEMA_ID = (
    "palette.coordinate_successor_authority"
)
COORDINATE_SUCCESSOR_AUTHORITY_SCHEMA_VERSION = 1
COORDINATE_SUCCESSOR_POLICY = (
    "immutable_source_authority_exact_payload_coordinate_republication_v1"
)

KEYPOINT_COORDINATE_SUCCESSOR_KIND = "raw_keypoint_coordinate_successor"
REFINED_SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND = (
    "refined_subject_mask_coordinate_successor"
)
SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND = "raw_subject_mask_coordinate_successor"
COORDINATE_SUCCESSOR_KINDS = frozenset(
    {
        KEYPOINT_COORDINATE_SUCCESSOR_KIND,
        SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
        REFINED_SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
    }
)


class CoordinateSuccessorAuthorityError(ValueError):
    """Raised when a coordinate-successor authority is absent or stale."""


def _sha256(value: object, *, name: str) -> str:
    result = str(value or "")
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise CoordinateSuccessorAuthorityError(
            f"{name} must be lowercase hexadecimal SHA-256."
        )
    return result


def _run_path(value: object, *, family: str, name: str) -> str:
    result = str(value or "")
    parts = result.split("/")
    if (
        len(parts) != 2
        or parts[0] != family
        or not parts[1]
        or parts[1] in {".", ".."}
    ):
        raise CoordinateSuccessorAuthorityError(
            f"{name} must name one exact {family}/<run> child."
        )
    return result


def build_coordinate_successor_authority(
    *,
    kind: str,
    source_family: str,
    source_run_path: str,
    source_manifest: Mapping[str, Any],
    source_authority_kind: str,
    source_authority: Mapping[str, Any],
    successor_family: str,
    successor_run_path: str,
    payload_equivalence: Mapping[str, Any],
    coordinate_records: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build one strict, self-digested selector-ineligible authority record."""

    if kind not in COORDINATE_SUCCESSOR_KINDS:
        raise CoordinateSuccessorAuthorityError(
            f"Unsupported coordinate successor kind {kind!r}."
        )
    source_path = _run_path(
        source_run_path, family=source_family, name="source_run_path"
    )
    successor_path = _run_path(
        successor_run_path, family=successor_family, name="successor_run_path"
    )
    if source_path == successor_path:
        raise CoordinateSuccessorAuthorityError(
            "Coordinate successor cannot replace its immutable source run."
        )
    if not isinstance(source_manifest, Mapping):
        raise CoordinateSuccessorAuthorityError("Source manifest must be an object.")
    source_payload = source_manifest.get("payload")
    logical = (
        source_payload.get("logical_content")
        if isinstance(source_payload, Mapping)
        else None
    )
    logical_digest = logical.get("digest") if isinstance(logical, Mapping) else None
    source = {
        "family": source_family,
        "run_path": source_path,
        "manifest_schema_id": source_manifest.get("schema_id"),
        "manifest_schema_version": source_manifest.get("schema_version"),
        "manifest_payload_digest": _sha256(
            source_manifest.get("payload_digest"),
            name="source manifest payload digest",
        ),
        "manifest_document_digest": canonical_json_sha256(source_manifest),
        "logical_content_digest": _sha256(
            logical_digest, name="source logical-content digest"
        ),
    }
    if not isinstance(source_authority_kind, str) or not source_authority_kind:
        raise CoordinateSuccessorAuthorityError(
            "source_authority_kind must be nonempty."
        )
    if not isinstance(source_authority, Mapping):
        raise CoordinateSuccessorAuthorityError(
            "source_authority must be an exact object."
        )
    if not isinstance(payload_equivalence, Mapping):
        raise CoordinateSuccessorAuthorityError(
            "payload_equivalence must be an exact object."
        )
    if not isinstance(coordinate_records, Mapping) or not coordinate_records:
        raise CoordinateSuccessorAuthorityError(
            "coordinate_records must bind one or more persisted records."
        )
    normalized_records: dict[str, dict[str, str]] = {}
    for name in sorted(coordinate_records):
        pointer = coordinate_records[name]
        if not isinstance(pointer, Mapping) or set(pointer) != {
            "record_ref",
            "record_sha256",
        }:
            raise CoordinateSuccessorAuthorityError(
                f"Coordinate record {name!r} is not one exact record pointer."
            )
        normalized_records[str(name)] = {
            "record_ref": str(pointer["record_ref"]),
            "record_sha256": _sha256(
                pointer["record_sha256"], name=f"coordinate record {name!r}"
            ),
        }
    payload = {
        "kind": kind,
        "policy": COORDINATE_SUCCESSOR_POLICY,
        "source": source,
        "source_authority": {
            "kind": source_authority_kind,
            "record_sha256": canonical_json_sha256(source_authority),
            "record": copy.deepcopy(dict(source_authority)),
        },
        "successor": {
            "family": successor_family,
            "run_path": successor_path,
            "coordinate_contract": "canonical_v2",
            "stage_selector_eligible": False,
            "selector_activation": "deferred_separate_reviewed_change",
        },
        "payload_equivalence": copy.deepcopy(dict(payload_equivalence)),
        "coordinate_records": normalized_records,
        "production_state_changes": [],
    }
    return {
        "schema_id": COORDINATE_SUCCESSOR_AUTHORITY_SCHEMA_ID,
        "schema_version": COORDINATE_SUCCESSOR_AUTHORITY_SCHEMA_VERSION,
        "digest_algorithm": "sha256_canonical_json_v1",
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def validate_coordinate_successor_authority(
    value: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    expected_successor_run_path: str | None = None,
) -> tuple[str, ...]:
    """Validate the strict JSON envelope without trusting a live run."""

    errors: list[str] = []
    if not isinstance(value, Mapping) or set(value) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        return ("coordinate successor authority envelope is not exact",)
    if (
        value.get("schema_id") != COORDINATE_SUCCESSOR_AUTHORITY_SCHEMA_ID
        or value.get("schema_version")
        != COORDINATE_SUCCESSOR_AUTHORITY_SCHEMA_VERSION
        or value.get("digest_algorithm") != "sha256_canonical_json_v1"
    ):
        errors.append("coordinate successor authority schema header differs")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "coordinate successor authority payload is absent")
    try:
        if value.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("coordinate successor authority payload digest differs")
    except (TypeError, ValueError) as exc:
        errors.append(f"coordinate successor authority is not strict JSON: {exc}")
    expected_payload_fields = {
        "kind",
        "policy",
        "source",
        "source_authority",
        "successor",
        "payload_equivalence",
        "coordinate_records",
        "production_state_changes",
    }
    if set(payload) != expected_payload_fields:
        errors.append("coordinate successor authority payload fields are not exact")
    kind = payload.get("kind")
    if kind not in COORDINATE_SUCCESSOR_KINDS:
        errors.append("coordinate successor kind is unsupported")
    if expected_kind is not None and kind != expected_kind:
        errors.append("coordinate successor kind differs from the requested source")
    if payload.get("policy") != COORDINATE_SUCCESSOR_POLICY:
        errors.append("coordinate successor policy differs")
    if payload.get("production_state_changes") != []:
        errors.append("coordinate successor reports production-state changes")
    successor = payload.get("successor")
    if not isinstance(successor, Mapping) or set(successor) != {
        "family",
        "run_path",
        "coordinate_contract",
        "stage_selector_eligible",
        "selector_activation",
    }:
        errors.append("coordinate successor target binding is malformed")
    else:
        family = successor.get("family")
        try:
            path = _run_path(
                successor.get("run_path"),
                family=str(family or ""),
                name="successor run_path",
            )
        except CoordinateSuccessorAuthorityError as exc:
            errors.append(str(exc))
            path = None
        if expected_successor_run_path is not None and path != expected_successor_run_path:
            errors.append("coordinate successor target run path differs")
        if (
            successor.get("coordinate_contract") != "canonical_v2"
            or successor.get("stage_selector_eligible") is not False
            or successor.get("selector_activation")
            != "deferred_separate_reviewed_change"
        ):
            errors.append("coordinate successor target lifecycle is invalid")
    source = payload.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "family",
        "run_path",
        "manifest_schema_id",
        "manifest_schema_version",
        "manifest_payload_digest",
        "manifest_document_digest",
        "logical_content_digest",
    }:
        errors.append("coordinate successor source binding is malformed")
    else:
        for name in (
            "manifest_payload_digest",
            "manifest_document_digest",
            "logical_content_digest",
        ):
            try:
                _sha256(source.get(name), name=f"source {name}")
            except CoordinateSuccessorAuthorityError as exc:
                errors.append(str(exc))
    source_authority = payload.get("source_authority")
    if not isinstance(source_authority, Mapping) or set(source_authority) != {
        "kind",
        "record_sha256",
        "record",
    }:
        errors.append("coordinate successor source authority is malformed")
    else:
        record = source_authority.get("record")
        if not isinstance(source_authority.get("kind"), str) or not source_authority.get(
            "kind"
        ):
            errors.append("coordinate successor source authority kind is absent")
        if not isinstance(record, Mapping):
            errors.append("coordinate successor source authority record is absent")
        else:
            try:
                if source_authority.get("record_sha256") != canonical_json_sha256(
                    record
                ):
                    errors.append("coordinate successor source authority digest differs")
            except (TypeError, ValueError) as exc:
                errors.append(f"coordinate successor source authority is invalid: {exc}")
    records = payload.get("coordinate_records")
    if not isinstance(records, Mapping) or not records:
        errors.append("coordinate successor record pointers are absent")
    else:
        for name, pointer in records.items():
            if not isinstance(pointer, Mapping) or set(pointer) != {
                "record_ref",
                "record_sha256",
            }:
                errors.append(f"coordinate successor pointer {name!r} is malformed")
                continue
            try:
                _sha256(pointer.get("record_sha256"), name=f"record {name!r}")
            except CoordinateSuccessorAuthorityError as exc:
                errors.append(str(exc))
    if not isinstance(payload.get("payload_equivalence"), Mapping):
        errors.append("coordinate successor payload-equivalence evidence is absent")
    return tuple(errors)


def stamp_coordinate_successor_authority(
    run: Any,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist one authority and its whole-document digest on a running child."""

    errors = validate_coordinate_successor_authority(
        authority,
        expected_successor_run_path=str(getattr(run, "path", "")).strip("/"),
    )
    if errors:
        raise CoordinateSuccessorAuthorityError(
            "Invalid coordinate successor authority: " + "; ".join(errors)
        )
    document = copy.deepcopy(dict(authority))
    digest = canonical_json_sha256(document)
    run.attrs[COORDINATE_SUCCESSOR_AUTHORITY_ATTR] = document
    run.attrs[COORDINATE_SUCCESSOR_AUTHORITY_DIGEST_ATTR] = digest
    if (
        run.attrs.get(COORDINATE_SUCCESSOR_AUTHORITY_ATTR) != document
        or run.attrs.get(COORDINATE_SUCCESSOR_AUTHORITY_DIGEST_ATTR) != digest
    ):
        raise CoordinateSuccessorAuthorityError(
            "Coordinate successor authority did not persist exactly."
        )
    return document


def load_coordinate_successor_authority(
    run: Any,
    *,
    expected_kind: str,
    expected_successor_run_path: str,
) -> dict[str, Any]:
    """Load an authority and verify every retained coordinate-record pointer."""

    value = getattr(run, "attrs", {}).get(COORDINATE_SUCCESSOR_AUTHORITY_ATTR)
    if not isinstance(value, Mapping):
        raise CoordinateSuccessorAuthorityError(
            "Coordinate successor authority is absent."
        )
    errors = validate_coordinate_successor_authority(
        value,
        expected_kind=expected_kind,
        expected_successor_run_path=expected_successor_run_path,
    )
    if errors:
        raise CoordinateSuccessorAuthorityError("; ".join(errors))
    document = copy.deepcopy(dict(value))
    if getattr(run, "attrs", {}).get(
        COORDINATE_SUCCESSOR_AUTHORITY_DIGEST_ATTR
    ) != canonical_json_sha256(document):
        raise CoordinateSuccessorAuthorityError(
            "Coordinate successor whole-document digest differs."
        )
    records = document["payload"]["coordinate_records"]
    for name, pointer in records.items():
        record_ref = str(pointer["record_ref"])
        if record_ref.count("@") != 1:
            raise CoordinateSuccessorAuthorityError(
                f"Coordinate successor pointer {name!r} is not canonical."
            )
        node_path, attr_name = record_ref.split("@", 1)
        run_path = str(getattr(run, "path", "")).strip("/")
        normalized_node = node_path.strip("/")
        if normalized_node != run_path and not normalized_node.startswith(
            f"{run_path}/"
        ):
            raise CoordinateSuccessorAuthorityError(
                f"Coordinate successor pointer {name!r} leaves the successor run."
            )
        relative = normalized_node[len(run_path) :].strip("/")
        try:
            node = run if not relative else run[relative]
            bound = bind_persisted_coordinate_record(node, attr_name=attr_name)
        except Exception as exc:
            raise CoordinateSuccessorAuthorityError(
                f"Coordinate successor record {name!r} is invalid: {exc}"
            ) from exc
        if (
            bound.record_ref != record_ref
            or bound.record_sha256 != pointer["record_sha256"]
        ):
            raise CoordinateSuccessorAuthorityError(
                f"Coordinate successor record {name!r} is stale."
            )
    return document


__all__ = [
    "COORDINATE_SUCCESSOR_AUTHORITY_ATTR",
    "COORDINATE_SUCCESSOR_AUTHORITY_DIGEST_ATTR",
    "COORDINATE_SUCCESSOR_AUTHORITY_SCHEMA_ID",
    "COORDINATE_SUCCESSOR_AUTHORITY_SCHEMA_VERSION",
    "COORDINATE_SUCCESSOR_KINDS",
    "COORDINATE_SUCCESSOR_POLICY",
    "CoordinateSuccessorAuthorityError",
    "KEYPOINT_COORDINATE_SUCCESSOR_KIND",
    "REFINED_SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND",
    "SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND",
    "build_coordinate_successor_authority",
    "load_coordinate_successor_authority",
    "stamp_coordinate_successor_authority",
    "validate_coordinate_successor_authority",
]
