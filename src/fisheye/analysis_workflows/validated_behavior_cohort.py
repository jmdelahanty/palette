"""Generic immutable cohort and bundle-set contracts for validated behavior.

These envelopes deliberately know nothing about a particular protocol or
scientific formula.  Source-specific adapters normalize a frozen roster and a
recording bundle into these contracts.  Exporters can then plan tables from a
closed membership axis and capability matrix without discovering Zarr runs or
teaching the publication layer about chasers, feeding, optomotor behavior, or
any other protocol family.

The module is pure contract machinery.  It never resolves selectors, mutates a
Zarr, writes a registry, or decides that a capability is scientifically valid.
Those decisions arrive as digest-bound adapter records.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from fisheye.shared.zarr.manifest_digest import canonical_json_sha256

MEMBERSHIP_SCHEMA_ID = "palette.analysis.validated_behavior_cohort_membership"
MEMBERSHIP_SCHEMA_VERSION = 1
MEMBERSHIP_STATUS = "complete_selector_ineligible_membership"
MEMBERSHIP_METHOD_ID = "normalized_digest_bound_source_membership_v1"

BUNDLE_SET_SCHEMA_ID = "palette.analysis.validated_behavior_bundle_set"
BUNDLE_SET_SCHEMA_VERSION = 1
BUNDLE_SET_STATUS = "complete_selector_ineligible_bundle_set"
BUNDLE_SET_METHOD_ID = "membership_closed_capability_composition_v1"

CAPABILITY_CONTRACT_SCHEMA_ID = "palette.analysis.behavior_capability_contract"
CAPABILITY_CONTRACT_SCHEMA_VERSION = 1

MEMBERSHIP_STATES = ("admitted", "excluded", "invalid", "unavailable")
MEMBERSHIP_REASON_CODES = MappingProxyType(
    {
        "admitted": frozenset({None}),
        "excluded": frozenset({"explicit_scientific_exclusion"}),
        "invalid": frozenset(
            {
                "invalid_semantic_selection",
                "invalid_source_authority",
                "identity_conflict",
            }
        ),
        "unavailable": frozenset(
            {
                "missing_required_authority",
                "missing_required_receipt",
            }
        ),
    }
)
BUNDLE_STATES = ("complete", "excluded", "invalid", "unavailable")
CAPABILITY_STATES = (
    "complete",
    "inapplicable",
    "invalid",
    "review_required",
    "stale",
    "unavailable",
)

ORDERING_POLICY = "lexicographic_canonical_analysis_zarr_then_recording_id_v1"
SELECTOR_PATH_POLICY = "forbid_selector_named_path_components_v1"
RELOCATION_POLICY = "new_audited_locator_successor_or_membership_generation_v1"

SAFETY = {
    "selector_eligible": False,
    "production_authority": False,
    "registry_update": False,
    "selector_activation": False,
    "source_mutation": False,
    "zarr_mutation": False,
}

MAX_MEMBERSHIP_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_SET_BYTES = 32 * 1024 * 1024

_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_SELECTOR_PARTS = frozenset(
    {
        "active",
        "active_run",
        "authoritative",
        "authoritative_run",
        "current",
        "current_run",
        "default",
        "default_run",
        "fallback",
        "latest",
        "latest_any",
        "latest_complete",
        "latest_pending",
        "selected",
        "selected_run",
    }
)


class ValidatedBehaviorCohortError(ValueError):
    """A normalized membership or bundle-set contract is not exact."""


def _fail(message: str) -> None:
    raise ValidatedBehaviorCohortError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{field} must be one object.")
    return value


def _list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field} must be one JSON array.")
    return value


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        _fail(f"{field} must be one non-empty exact string.")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field=field)


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if _DIGEST_RE.fullmatch(result) is None:
        _fail(f"{field} must be one lowercase SHA-256 digest.")
    return result


def _optional_digest(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _digest(value, field=field)


def _commit(value: object) -> str:
    result = _text(value, field="software_authority.commit")
    if _COMMIT_RE.fullmatch(result) is None:
        _fail("software_authority.commit must be one full lowercase Git object ID.")
    return result


def _safe_id(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if len(result) > 160 or _SAFE_ID_RE.fullmatch(result) is None:
        _fail(f"{field} must be one portable identifier.")
    return result


def _positive_int(value: object, *, field: str, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        bound = "non-negative" if allow_zero else "positive"
        _fail(f"{field} must be one {bound} integer.")
    return value


def _utc_timestamp(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidatedBehaviorCohortError(
            f"{field} must be one ISO-8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{field} must include an explicit UTC offset.")
    if parsed.utcoffset().total_seconds() != 0:
        _fail(f"{field} must be expressed in UTC.")
    return result


def _canonical_absolute_path(
    value: object,
    *,
    field: str,
    root: Path | None = None,
    forbid_selectors: bool = False,
) -> Path:
    raw = str(value) if isinstance(value, Path) else _text(value, field=field)
    path = Path(raw)
    if not path.is_absolute() or "\\" in raw:
        _fail(f"{field} must be one canonical absolute POSIX path.")
    resolved = path.expanduser().resolve(strict=False)
    if str(resolved) != raw:
        _fail(f"{field} must already be canonical.")
    if root is not None:
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValidatedBehaviorCohortError(
                f"{field} escapes its declared root."
            ) from exc
    if forbid_selectors and any(
        part.casefold() in _SELECTOR_PARTS for part in PurePosixPath(raw).parts
    ):
        _fail(f"{field} contains a forbidden selector-named path component.")
    return resolved


def _strict_json_size(value: object, *, limit: int, field: str) -> None:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidatedBehaviorCohortError(
            f"{field} is not strict JSON: {exc}"
        ) from exc
    if len(encoded) > limit:
        _fail(f"{field} exceeds its {limit}-byte contract limit.")


def _sorted_unique_texts(value: object, *, field: str, nonempty: bool) -> list[str]:
    items = _list(value, field=field)
    normalized = [_text(item, field=f"{field}[]") for item in items]
    if len(set(normalized)) != len(normalized) or normalized != sorted(normalized):
        _fail(f"{field} must be unique and lexicographically sorted.")
    if nonempty and not normalized:
        _fail(f"{field} must not be empty.")
    return normalized


def _policy_envelope(value: object, *, field: str) -> dict[str, Any]:
    envelope = _mapping(value, field=field)
    if set(envelope) != {"record", "sha256"}:
        _fail(f"{field} must contain exactly record and sha256.")
    record = _mapping(envelope.get("record"), field=f"{field}.record")
    if not record:
        _fail(f"{field}.record must not be empty.")
    observed = canonical_json_sha256(_plain(record))
    if _digest(envelope.get("sha256"), field=f"{field}.sha256") != observed:
        _fail(f"{field} digest is stale.")
    return {"record": _plain(record), "sha256": observed}


def policy_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    """Create one canonical digest envelope for a caller-owned policy record."""

    normalized = _plain(_mapping(record, field="policy record"))
    if not normalized:
        _fail("policy record must not be empty.")
    _strict_json_size(normalized, limit=1_048_576, field="policy record")
    return {"record": normalized, "sha256": canonical_json_sha256(normalized)}


def _validate_software(value: object) -> dict[str, str]:
    software = _mapping(value, field="software_authority")
    if set(software) != {"repository", "commit"}:
        _fail("software_authority field set is inexact.")
    if software.get("repository") != "palette":
        _fail("software_authority.repository must be 'palette'.")
    return {"repository": "palette", "commit": _commit(software.get("commit"))}


def _validate_source_membership(value: object) -> dict[str, Any]:
    source = _mapping(value, field="source_membership")
    required = {
        "adapter_id",
        "schema_id",
        "schema_version",
        "profile",
        "path",
        "file_sha256",
        "record_sha256",
        "member_count",
        "source_members_sha256",
    }
    if set(source) != required:
        _fail("source_membership field set is inexact.")
    path = _canonical_absolute_path(
        source.get("path"),
        field="source_membership.path",
        forbid_selectors=True,
    )
    schema_version = _positive_int(
        source.get("schema_version"), field="source_membership.schema_version"
    )
    count = _positive_int(
        source.get("member_count"), field="source_membership.member_count"
    )
    return {
        "adapter_id": _safe_id(
            source.get("adapter_id"), field="source_membership.adapter_id"
        ),
        "schema_id": _text(
            source.get("schema_id"), field="source_membership.schema_id"
        ),
        "schema_version": schema_version,
        "profile": _safe_id(source.get("profile"), field="source_membership.profile"),
        "path": str(path),
        "file_sha256": _digest(
            source.get("file_sha256"), field="source_membership.file_sha256"
        ),
        "record_sha256": _digest(
            source.get("record_sha256"), field="source_membership.record_sha256"
        ),
        "member_count": count,
        "source_members_sha256": _digest(
            source.get("source_members_sha256"),
            field="source_membership.source_members_sha256",
        ),
    }


def _validate_locator_policy(value: object) -> dict[str, str]:
    policy = _mapping(value, field="locator_policy")
    if set(policy) != {
        "analysis_zarr_root",
        "admission_receipt_root",
        "selector_path_policy",
        "relocation_policy",
    }:
        _fail("locator_policy field set is inexact.")
    analysis_root = _canonical_absolute_path(
        policy.get("analysis_zarr_root"), field="locator_policy.analysis_zarr_root"
    )
    receipt_root = _canonical_absolute_path(
        policy.get("admission_receipt_root"),
        field="locator_policy.admission_receipt_root",
    )
    if policy.get("selector_path_policy") != SELECTOR_PATH_POLICY:
        _fail("locator_policy.selector_path_policy is unsupported.")
    if policy.get("relocation_policy") != RELOCATION_POLICY:
        _fail("locator_policy.relocation_policy is unsupported.")
    return {
        "analysis_zarr_root": str(analysis_root),
        "admission_receipt_root": str(receipt_root),
        "selector_path_policy": SELECTOR_PATH_POLICY,
        "relocation_policy": RELOCATION_POLICY,
    }


def _validate_disposition_evidence(value: object, *, field: str) -> dict[str, Any]:
    evidence = _mapping(value, field=field)
    if set(evidence) != {
        "evidence_type",
        "detail",
        "path",
        "file_sha256",
        "record_sha256",
    }:
        _fail(f"{field} field set is inexact.")
    detail = _optional_text(evidence.get("detail"), field=f"{field}.detail")
    if detail is not None and len(detail.encode("utf-8")) > 2048:
        _fail(f"{field}.detail exceeds 2048 UTF-8 bytes.")
    raw_path = evidence.get("path")
    path = (
        None
        if raw_path is None
        else str(_canonical_absolute_path(raw_path, field=f"{field}.path"))
    )
    file_sha = _optional_digest(
        evidence.get("file_sha256"), field=f"{field}.file_sha256"
    )
    record_sha = _optional_digest(
        evidence.get("record_sha256"), field=f"{field}.record_sha256"
    )
    if (path is None) != (file_sha is None):
        _fail(f"{field}.path and file_sha256 must be present together.")
    if record_sha is not None and path is None:
        _fail(f"{field}.record_sha256 requires a bound path.")
    return {
        "evidence_type": _safe_id(
            evidence.get("evidence_type"), field=f"{field}.evidence_type"
        ),
        "detail": detail,
        "path": path,
        "file_sha256": file_sha,
        "record_sha256": record_sha,
    }


def _validate_receipt_binding(
    value: object,
    *,
    field: str,
    root: Path,
) -> dict[str, Any]:
    receipt = _mapping(value, field=field)
    if set(receipt) != {
        "role",
        "path",
        "file_sha256",
        "record_sha256",
        "schema_id",
        "schema_version",
    }:
        _fail(f"{field} field set is inexact.")
    path = _canonical_absolute_path(
        receipt.get("path"), field=f"{field}.path", root=root, forbid_selectors=True
    )
    return {
        "role": _safe_id(receipt.get("role"), field=f"{field}.role"),
        "path": str(path),
        "file_sha256": _digest(
            receipt.get("file_sha256"), field=f"{field}.file_sha256"
        ),
        "record_sha256": _digest(
            receipt.get("record_sha256"), field=f"{field}.record_sha256"
        ),
        "schema_id": _text(receipt.get("schema_id"), field=f"{field}.schema_id"),
        "schema_version": _positive_int(
            receipt.get("schema_version"), field=f"{field}.schema_version"
        ),
    }


def _validate_membership_member(
    value: object,
    *,
    ordinal: int,
    analysis_root: Path,
    receipt_root: Path,
    analysis_unit_policy: Mapping[str, Any],
    acquisition_batch_policy: Mapping[str, Any],
) -> dict[str, Any]:
    member = _mapping(value, field=f"members[{ordinal - 1}]")
    required = {
        "ordinal",
        "source_ordinal",
        "dataset_id",
        "recording_id",
        "analysis_zarr",
        "protocol_names",
        "protocol_hashes",
        "source_member_sha256",
        "source_subject_ids",
        "source_subject_identity_status",
        "acquisition_batch_id",
        "acquisition_batch_identity_status",
        "analysis_unit_kind",
        "analysis_unit_id",
        "membership_state",
        "reason_code",
        "disposition_evidence",
        "admission_receipts",
        "member_sha256",
    }
    if set(member) != required:
        _fail(f"members[{ordinal - 1}] field set is inexact.")
    persisted = _digest(
        member.get("member_sha256"), field=f"members[{ordinal - 1}].member_sha256"
    )
    body = {key: _plain(item) for key, item in member.items() if key != "member_sha256"}
    if canonical_json_sha256(body) != persisted:
        _fail(f"members[{ordinal - 1}] digest is stale.")
    if member.get("ordinal") != ordinal:
        _fail("Membership ordinals must be a contiguous one-based axis.")
    source_ordinal = _positive_int(
        member.get("source_ordinal"), field=f"members[{ordinal - 1}].source_ordinal"
    )
    dataset_id = _text(
        member.get("dataset_id"), field=f"members[{ordinal - 1}].dataset_id"
    )
    recording_id = _text(
        member.get("recording_id"), field=f"members[{ordinal - 1}].recording_id"
    )
    analysis_zarr = _canonical_absolute_path(
        member.get("analysis_zarr"),
        field=f"members[{ordinal - 1}].analysis_zarr",
        root=analysis_root,
    )
    protocol_names = _sorted_unique_texts(
        member.get("protocol_names"),
        field=f"members[{ordinal - 1}].protocol_names",
        nonempty=False,
    )
    protocol_hashes = _sorted_unique_texts(
        member.get("protocol_hashes"),
        field=f"members[{ordinal - 1}].protocol_hashes",
        nonempty=False,
    )
    if bool(protocol_names) != bool(protocol_hashes):
        _fail(
            f"members[{ordinal - 1}] protocol names and hashes must be "
            "coherently present or absent."
        )
    for index, item in enumerate(protocol_hashes):
        _digest(item, field=f"members[{ordinal - 1}].protocol_hashes[{index}]")
    source_subject_ids = _sorted_unique_texts(
        member.get("source_subject_ids"),
        field=f"members[{ordinal - 1}].source_subject_ids",
        nonempty=False,
    )
    batch_id = _optional_text(
        member.get("acquisition_batch_id"),
        field=f"members[{ordinal - 1}].acquisition_batch_id",
    )
    batch_status = _text(
        member.get("acquisition_batch_identity_status"),
        field=f"members[{ordinal - 1}].acquisition_batch_identity_status",
    )
    batch_policy_record = _mapping(
        acquisition_batch_policy.get("record"),
        field="acquisition_batch_policy.record",
    )
    expected_batch_status = (
        batch_policy_record.get("missing_identity_status")
        if batch_id is None
        else batch_policy_record.get("authoritative_identity_status")
    )
    if batch_status != expected_batch_status:
        _fail("Acquisition-batch value and identity status disagree.")
    state = member.get("membership_state")
    if state not in MEMBERSHIP_STATES:
        _fail(f"members[{ordinal - 1}].membership_state is invalid.")
    reason = member.get("reason_code")
    if reason not in MEMBERSHIP_REASON_CODES[state]:
        _fail(f"members[{ordinal - 1}].reason_code is invalid for {state!r}.")
    evidence = _validate_disposition_evidence(
        member.get("disposition_evidence"),
        field=f"members[{ordinal - 1}].disposition_evidence",
    )
    receipts = [
        _validate_receipt_binding(
            item,
            field=f"members[{ordinal - 1}].admission_receipts[{index}]",
            root=receipt_root,
        )
        for index, item in enumerate(
            _list(
                member.get("admission_receipts"),
                field=f"members[{ordinal - 1}].admission_receipts",
            )
        )
    ]
    receipt_order = [(item["role"], item["path"]) for item in receipts]
    if len(set(receipt_order)) != len(receipt_order) or receipt_order != sorted(
        receipt_order
    ):
        _fail("Admission receipts must be uniquely and deterministically ordered.")
    if (state == "admitted") != bool(receipts):
        _fail(
            "Only admitted members may carry admission receipts, and they require one."
        )
    unit_kind = _text(
        member.get("analysis_unit_kind"),
        field=f"members[{ordinal - 1}].analysis_unit_kind",
    )
    unit_id = _text(
        member.get("analysis_unit_id"),
        field=f"members[{ordinal - 1}].analysis_unit_id",
    )
    policy_record = _mapping(
        analysis_unit_policy.get("record"), field="analysis_unit_policy.record"
    )
    if unit_kind != policy_record.get("analysis_unit_kind"):
        _fail("Member analysis-unit kind differs from the bound policy.")
    member_id_field = policy_record.get("member_id_field")
    expected_unit_id = {"recording_id": recording_id, "dataset_id": dataset_id}.get(
        member_id_field
    )
    if expected_unit_id is None or unit_id != expected_unit_id:
        _fail("Member analysis-unit ID differs from the bound policy.")
    if unit_id in source_subject_ids:
        _fail("A source subject identity cannot silently become the analysis-unit ID.")
    return {
        "ordinal": ordinal,
        "source_ordinal": source_ordinal,
        "dataset_id": dataset_id,
        "recording_id": recording_id,
        "analysis_zarr": str(analysis_zarr),
        "protocol_names": protocol_names,
        "protocol_hashes": protocol_hashes,
        "source_member_sha256": _digest(
            member.get("source_member_sha256"),
            field=f"members[{ordinal - 1}].source_member_sha256",
        ),
        "source_subject_ids": source_subject_ids,
        "source_subject_identity_status": _safe_id(
            member.get("source_subject_identity_status"),
            field=f"members[{ordinal - 1}].source_subject_identity_status",
        ),
        "acquisition_batch_id": batch_id,
        "acquisition_batch_identity_status": batch_status,
        "analysis_unit_kind": unit_kind,
        "analysis_unit_id": unit_id,
        "membership_state": state,
        "reason_code": reason,
        "disposition_evidence": evidence,
        "admission_receipts": receipts,
        "member_sha256": persisted,
    }


def build_validated_behavior_cohort_membership(
    *,
    membership_id: str,
    source_membership: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    analysis_zarr_root: str | Path,
    admission_receipt_root: str | Path,
    analysis_unit_policy: Mapping[str, Any],
    acquisition_batch_policy: Mapping[str, Any],
    temporal_alignment_policy: Mapping[str, Any],
    palette_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Build a protocol-independent normalized cohort membership manifest."""

    normalized_source = _validate_source_membership(source_membership)
    analysis_root = Path(analysis_zarr_root).expanduser().resolve(strict=False)
    receipt_root = Path(admission_receipt_root).expanduser().resolve(strict=False)
    raw_members = [_plain(_mapping(item, field="member")) for item in members]
    if not raw_members:
        _fail("Membership must contain at least one member.")
    raw_members.sort(
        key=lambda item: (
            str(item.get("analysis_zarr") or ""),
            str(item.get("recording_id") or ""),
            str(item.get("dataset_id") or ""),
        )
    )
    sealed_members: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_members, start=1):
        body = {**raw, "ordinal": ordinal}
        body.pop("member_sha256", None)
        sealed_members.append({**body, "member_sha256": canonical_json_sha256(body)})
    body = {
        "schema_id": MEMBERSHIP_SCHEMA_ID,
        "schema_version": MEMBERSHIP_SCHEMA_VERSION,
        "method_id": MEMBERSHIP_METHOD_ID,
        "status": MEMBERSHIP_STATUS,
        "membership_id": _safe_id(membership_id, field="membership_id"),
        "source_membership": normalized_source,
        "ordering_policy": ORDERING_POLICY,
        "locator_policy": {
            "analysis_zarr_root": str(analysis_root),
            "admission_receipt_root": str(receipt_root),
            "selector_path_policy": SELECTOR_PATH_POLICY,
            "relocation_policy": RELOCATION_POLICY,
        },
        "analysis_unit_policy": _policy_envelope(
            analysis_unit_policy, field="analysis_unit_policy"
        ),
        "acquisition_batch_policy": _policy_envelope(
            acquisition_batch_policy, field="acquisition_batch_policy"
        ),
        "temporal_alignment_policy": _policy_envelope(
            temporal_alignment_policy, field="temporal_alignment_policy"
        ),
        "member_count": len(sealed_members),
        "state_counts": {
            state: sum(item["membership_state"] == state for item in sealed_members)
            for state in MEMBERSHIP_STATES
        },
        "members": sealed_members,
        "members_sha256": canonical_json_sha256(sealed_members),
        "software_authority": {
            "repository": "palette",
            "commit": _commit(palette_commit),
        },
        "created_at_utc": _utc_timestamp(created_at_utc, field="created_at_utc"),
        "safety": SAFETY,
    }
    result = {**body, "record_sha256": canonical_json_sha256(body)}
    validate_validated_behavior_cohort_membership(result)
    return result


def validate_validated_behavior_cohort_membership(value: object) -> Mapping[str, Any]:
    """Validate the generic membership envelope without resolving its adapter."""

    membership = _plain(_mapping(value, field="membership"))
    _strict_json_size(membership, limit=MAX_MEMBERSHIP_BYTES, field="membership")
    persisted = membership.pop("record_sha256", None)
    if _digest(persisted, field="record_sha256") != canonical_json_sha256(membership):
        _fail("Membership record digest is stale.")
    required = {
        "schema_id",
        "schema_version",
        "method_id",
        "status",
        "membership_id",
        "source_membership",
        "ordering_policy",
        "locator_policy",
        "analysis_unit_policy",
        "acquisition_batch_policy",
        "temporal_alignment_policy",
        "member_count",
        "state_counts",
        "members",
        "members_sha256",
        "software_authority",
        "created_at_utc",
        "safety",
    }
    if set(membership) != required:
        _fail("Membership field set is inexact.")
    if (
        membership.get("schema_id") != MEMBERSHIP_SCHEMA_ID
        or membership.get("schema_version") != MEMBERSHIP_SCHEMA_VERSION
        or membership.get("method_id") != MEMBERSHIP_METHOD_ID
        or membership.get("status") != MEMBERSHIP_STATUS
        or membership.get("ordering_policy") != ORDERING_POLICY
        or membership.get("safety") != SAFETY
    ):
        _fail("Membership identity, method, ordering, status, or safety is invalid.")
    _safe_id(membership.get("membership_id"), field="membership_id")
    source = _validate_source_membership(membership.get("source_membership"))
    locator = _validate_locator_policy(membership.get("locator_policy"))
    unit_policy = _policy_envelope(
        membership.get("analysis_unit_policy"), field="analysis_unit_policy"
    )
    batch_policy = _policy_envelope(
        membership.get("acquisition_batch_policy"), field="acquisition_batch_policy"
    )
    temporal_policy = _policy_envelope(
        membership.get("temporal_alignment_policy"), field="temporal_alignment_policy"
    )
    members_raw = _list(membership.get("members"), field="members")
    count = _positive_int(membership.get("member_count"), field="member_count")
    if count != len(members_raw) or source["member_count"] != count:
        _fail("Membership count disagrees with its source or member roster.")
    analysis_root = Path(locator["analysis_zarr_root"])
    receipt_root = Path(locator["admission_receipt_root"])
    members = [
        _validate_membership_member(
            raw,
            ordinal=index,
            analysis_root=analysis_root,
            receipt_root=receipt_root,
            analysis_unit_policy=unit_policy,
            acquisition_batch_policy=batch_policy,
        )
        for index, raw in enumerate(members_raw, start=1)
    ]
    observed_order = [
        (item["analysis_zarr"], item["recording_id"], item["dataset_id"])
        for item in members
    ]
    if observed_order != sorted(observed_order):
        _fail("Membership members are not in deterministic normalized order.")
    for field in ("recording_id", "dataset_id", "analysis_zarr", "analysis_unit_id"):
        values = [item[field] for item in members]
        if len(set(values)) != len(values):
            _fail(f"Membership contains a duplicate {field}.")
    source_ordinals = [item["source_ordinal"] for item in members]
    if len(set(source_ordinals)) != len(source_ordinals):
        _fail("Membership contains a duplicate source ordinal.")
    if canonical_json_sha256(members) != _digest(
        membership.get("members_sha256"), field="members_sha256"
    ):
        _fail("Membership aggregate member digest is stale.")
    state_counts = _mapping(membership.get("state_counts"), field="state_counts")
    expected_counts = {
        state: sum(item["membership_state"] == state for item in members)
        for state in MEMBERSHIP_STATES
    }
    if (
        set(state_counts) != set(MEMBERSHIP_STATES)
        or _plain(state_counts) != expected_counts
    ):
        _fail("Membership state counts are stale or incomplete.")
    _validate_software(membership.get("software_authority"))
    _utc_timestamp(membership.get("created_at_utc"), field="created_at_utc")
    normalized = {
        **membership,
        "source_membership": source,
        "locator_policy": locator,
        "analysis_unit_policy": unit_policy,
        "acquisition_batch_policy": batch_policy,
        "temporal_alignment_policy": temporal_policy,
        "members": members,
        "record_sha256": persisted,
    }
    return _freeze(normalized)


def _validate_capability_contract(value: object) -> dict[str, Any]:
    contract = _mapping(value, field="capability_contract")
    if set(contract) != {
        "schema_id",
        "schema_version",
        "profile_id",
        "keys",
        "states",
        "reason_codes_by_state",
        "record_sha256",
    }:
        _fail("capability_contract field set is inexact.")
    persisted = _digest(
        contract.get("record_sha256"), field="capability_contract.record_sha256"
    )
    body = {
        key: _plain(item) for key, item in contract.items() if key != "record_sha256"
    }
    if canonical_json_sha256(body) != persisted:
        _fail("Capability contract digest is stale.")
    if (
        contract.get("schema_id") != CAPABILITY_CONTRACT_SCHEMA_ID
        or contract.get("schema_version") != CAPABILITY_CONTRACT_SCHEMA_VERSION
    ):
        _fail("Capability contract schema is unsupported.")
    keys = _sorted_unique_texts(
        contract.get("keys"), field="capability_contract.keys", nonempty=True
    )
    states = _sorted_unique_texts(
        contract.get("states"), field="capability_contract.states", nonempty=True
    )
    if set(states) != set(CAPABILITY_STATES):
        _fail("Capability contract state vocabulary is unsupported.")
    reasons_raw = _mapping(
        contract.get("reason_codes_by_state"),
        field="capability_contract.reason_codes_by_state",
    )
    if set(reasons_raw) != set(states):
        _fail("Capability reason-code states are incomplete.")
    reasons: dict[str, list[str | None]] = {}
    for state in states:
        raw = _list(reasons_raw[state], field=f"capability reason codes for {state}")
        normalized: list[str | None] = []
        for item in raw:
            normalized.append(
                None if item is None else _safe_id(item, field="reason code")
            )

        def sort_key(item: str | None) -> str:
            return "" if item is None else item

        if len(set(normalized)) != len(normalized) or normalized != sorted(
            normalized, key=sort_key
        ):
            _fail(f"Capability reasons for {state!r} must be unique and sorted.")
        if (state == "complete" and normalized != [None]) or (
            state != "complete" and (not normalized or None in normalized)
        ):
            _fail(f"Capability reasons for {state!r} violate state semantics.")
        reasons[state] = normalized
    return {
        "schema_id": CAPABILITY_CONTRACT_SCHEMA_ID,
        "schema_version": CAPABILITY_CONTRACT_SCHEMA_VERSION,
        "profile_id": _safe_id(
            contract.get("profile_id"), field="capability_contract.profile_id"
        ),
        "keys": keys,
        "states": states,
        "reason_codes_by_state": reasons,
        "record_sha256": persisted,
    }


def build_capability_contract(
    *,
    profile_id: str,
    keys: Sequence[str],
    reason_codes_by_state: Mapping[str, Sequence[str | None]],
) -> dict[str, Any]:
    """Seal a profile-owned capability vocabulary for the generic bundle set."""

    body = {
        "schema_id": CAPABILITY_CONTRACT_SCHEMA_ID,
        "schema_version": CAPABILITY_CONTRACT_SCHEMA_VERSION,
        "profile_id": _safe_id(profile_id, field="profile_id"),
        "keys": sorted({_safe_id(item, field="capability key") for item in keys}),
        "states": sorted(CAPABILITY_STATES),
        "reason_codes_by_state": {
            state: sorted(
                set(reason_codes_by_state[state]),
                key=lambda item: "" if item is None else str(item),
            )
            for state in sorted(CAPABILITY_STATES)
        },
    }
    result = {**body, "record_sha256": canonical_json_sha256(body)}
    return _plain(_validate_capability_contract(result))


def _validate_capability(
    value: object,
    *,
    field: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    item = _mapping(value, field=field)
    if set(item) != {"state", "reason_code", "detail", "binding"}:
        _fail(f"{field} field set is inexact.")
    state = item.get("state")
    if state not in contract["states"]:
        _fail(f"{field}.state is invalid.")
    reason = item.get("reason_code")
    if reason not in contract["reason_codes_by_state"][state]:
        _fail(f"{field}.reason_code is invalid for state {state!r}.")
    detail = _optional_text(item.get("detail"), field=f"{field}.detail")
    if detail is not None and len(detail.encode("utf-8")) > 1024:
        _fail(f"{field}.detail exceeds 1024 UTF-8 bytes.")
    binding_raw = item.get("binding")
    if state == "complete":
        binding = _mapping(binding_raw, field=f"{field}.binding")
        if not binding:
            _fail(f"{field}.binding must not be empty for a complete capability.")
        binding_value: dict[str, Any] | None = _plain(binding)
    else:
        if binding_raw is not None:
            _fail(f"{field}.binding must be null for a non-complete capability.")
        binding_value = None
    return {
        "state": state,
        "reason_code": reason,
        "detail": detail,
        "binding": binding_value,
    }


def _validate_bundle_binding(
    value: object, *, field: str, bundle_root: Path
) -> dict[str, Any]:
    bundle = _mapping(value, field=field)
    if set(bundle) != {
        "adapter_id",
        "path",
        "file_sha256",
        "record_sha256",
        "schema_id",
        "schema_version",
        "method_id",
        "status",
        "receipt_bindings",
        "binding_inventory_sha256",
    }:
        _fail(f"{field} field set is inexact.")
    path = _canonical_absolute_path(
        bundle.get("path"),
        field=f"{field}.path",
        root=bundle_root,
        forbid_selectors=True,
    )
    receipts = [
        _validate_receipt_binding(
            item,
            field=f"{field}.receipt_bindings[{index}]",
            root=Path("/"),
        )
        for index, item in enumerate(
            _list(bundle.get("receipt_bindings"), field=f"{field}.receipt_bindings")
        )
    ]
    receipt_order = [(item["role"], item["path"]) for item in receipts]
    if len(set(receipt_order)) != len(receipt_order) or receipt_order != sorted(
        receipt_order
    ):
        _fail(f"{field}.receipt_bindings must be unique and sorted.")
    return {
        "adapter_id": _safe_id(bundle.get("adapter_id"), field=f"{field}.adapter_id"),
        "path": str(path),
        "file_sha256": _digest(bundle.get("file_sha256"), field=f"{field}.file_sha256"),
        "record_sha256": _digest(
            bundle.get("record_sha256"), field=f"{field}.record_sha256"
        ),
        "schema_id": _text(bundle.get("schema_id"), field=f"{field}.schema_id"),
        "schema_version": _positive_int(
            bundle.get("schema_version"), field=f"{field}.schema_version"
        ),
        "method_id": _safe_id(bundle.get("method_id"), field=f"{field}.method_id"),
        "status": _safe_id(bundle.get("status"), field=f"{field}.status"),
        "receipt_bindings": receipts,
        "binding_inventory_sha256": _digest(
            bundle.get("binding_inventory_sha256"),
            field=f"{field}.binding_inventory_sha256",
        ),
    }


def _validate_bundle_set_member(
    value: object,
    *,
    membership_member: Mapping[str, Any],
    capability_contract: Mapping[str, Any],
    bundle_root: Path,
) -> dict[str, Any]:
    ordinal = int(membership_member["ordinal"])
    field = f"members[{ordinal - 1}]"
    item = _mapping(value, field=field)
    required = {
        "ordinal",
        "membership_member_sha256",
        "recording_id",
        "analysis_zarr",
        "bundle_state",
        "reason_code",
        "bundle",
        "capabilities",
        "capabilities_sha256",
        "member_sha256",
    }
    if set(item) != required:
        _fail(f"{field} field set is inexact.")
    persisted = _digest(item.get("member_sha256"), field=f"{field}.member_sha256")
    body = {key: _plain(raw) for key, raw in item.items() if key != "member_sha256"}
    if canonical_json_sha256(body) != persisted:
        _fail(f"{field} digest is stale.")
    if (
        item.get("ordinal") != ordinal
        or item.get("membership_member_sha256") != membership_member["member_sha256"]
        or item.get("recording_id") != membership_member["recording_id"]
        or item.get("analysis_zarr") != membership_member["analysis_zarr"]
    ):
        _fail(f"{field} differs from its membership identity.")
    state = item.get("bundle_state")
    expected_state = {
        "admitted": "complete",
        "excluded": "excluded",
        "invalid": "invalid",
        "unavailable": "unavailable",
    }[membership_member["membership_state"]]
    if state != expected_state:
        _fail(f"{field}.bundle_state disagrees with membership.")
    reason = item.get("reason_code")
    expected_reason = None if state == "complete" else membership_member["reason_code"]
    if reason != expected_reason:
        _fail(f"{field}.reason_code disagrees with membership.")
    if state == "complete":
        bundle = _validate_bundle_binding(
            item.get("bundle"), field=f"{field}.bundle", bundle_root=bundle_root
        )
    else:
        if item.get("bundle") is not None:
            _fail(f"{field}.bundle must be null for a non-complete member.")
        bundle = None
    capabilities_raw = _mapping(item.get("capabilities"), field=f"{field}.capabilities")
    if set(capabilities_raw) != set(capability_contract["keys"]):
        _fail(f"{field}.capability roster is inexact.")
    capabilities = {
        key: _validate_capability(
            capabilities_raw[key],
            field=f"{field}.capabilities.{key}",
            contract=capability_contract,
        )
        for key in capability_contract["keys"]
    }
    if state != "complete" and any(
        capability["state"] == "complete" for capability in capabilities.values()
    ):
        _fail(f"{field} cannot expose a complete capability without a complete bundle.")
    capability_sha = _digest(
        item.get("capabilities_sha256"), field=f"{field}.capabilities_sha256"
    )
    if canonical_json_sha256(capabilities) != capability_sha:
        _fail(f"{field} capability digest is stale.")
    return {
        "ordinal": ordinal,
        "membership_member_sha256": membership_member["member_sha256"],
        "recording_id": membership_member["recording_id"],
        "analysis_zarr": membership_member["analysis_zarr"],
        "bundle_state": state,
        "reason_code": reason,
        "bundle": bundle,
        "capabilities": capabilities,
        "capabilities_sha256": capability_sha,
        "member_sha256": persisted,
    }


def _bundle_binding_projection(member: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ordinal": member["ordinal"],
        "membership_member_sha256": member["membership_member_sha256"],
        "recording_id": member["recording_id"],
        "analysis_zarr": member["analysis_zarr"],
        "bundle_state": member["bundle_state"],
        "reason_code": member["reason_code"],
        "bundle": _plain(member["bundle"]),
    }


def _capability_projection(member: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ordinal": member["ordinal"],
        "recording_id": member["recording_id"],
        "capabilities": _plain(member["capabilities"]),
        "capabilities_sha256": member["capabilities_sha256"],
    }


def build_validated_behavior_bundle_set(
    *,
    bundle_set_id: str,
    membership: Mapping[str, Any],
    membership_path: str | Path,
    membership_file_sha256: str,
    bundle_root: str | Path,
    bundle_profile: Mapping[str, Any],
    capability_contract: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    palette_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Build a generic closed bundle/capability matrix over one membership."""

    validated_membership = validate_validated_behavior_cohort_membership(membership)
    membership_file = _canonical_absolute_path(
        membership_path, field="membership.path", forbid_selectors=True
    )
    root = Path(bundle_root).expanduser().resolve(strict=False)
    contract = _validate_capability_contract(capability_contract)
    profile = _plain(_mapping(bundle_profile, field="bundle_profile"))
    if not profile:
        _fail("bundle_profile must not be empty.")
    membership_members = list(validated_membership["members"])
    by_recording: dict[str, Mapping[str, Any]] = {}
    for raw in members:
        record = _mapping(raw, field="bundle-set member")
        recording_id = _text(
            record.get("recording_id"), field="bundle-set recording_id"
        )
        if recording_id in by_recording:
            _fail("Bundle-set input contains a duplicate recording.")
        by_recording[recording_id] = record
    expected_ids = {str(item["recording_id"]) for item in membership_members}
    if set(by_recording) != expected_ids:
        _fail("Bundle-set input must contain exactly one record for every member.")
    sealed: list[dict[str, Any]] = []
    for membership_member in membership_members:
        raw = _plain(by_recording[str(membership_member["recording_id"])])
        raw["ordinal"] = membership_member["ordinal"]
        raw["membership_member_sha256"] = membership_member["member_sha256"]
        raw["analysis_zarr"] = membership_member["analysis_zarr"]
        capabilities = _plain(_mapping(raw.get("capabilities"), field="capabilities"))
        raw["capabilities_sha256"] = canonical_json_sha256(capabilities)
        raw.pop("member_sha256", None)
        sealed.append({**raw, "member_sha256": canonical_json_sha256(raw)})
    source_binding = {
        "path": str(membership_file),
        "file_sha256": _digest(membership_file_sha256, field="membership.file_sha256"),
        "record_sha256": str(validated_membership["record_sha256"]),
        "members_sha256": str(validated_membership["members_sha256"]),
        "member_count": int(validated_membership["member_count"]),
    }
    body = {
        "schema_id": BUNDLE_SET_SCHEMA_ID,
        "schema_version": BUNDLE_SET_SCHEMA_VERSION,
        "method_id": BUNDLE_SET_METHOD_ID,
        "status": BUNDLE_SET_STATUS,
        "bundle_set_id": _safe_id(bundle_set_id, field="bundle_set_id"),
        "membership": source_binding,
        "bundle_root": str(root),
        "bundle_profile": profile,
        "capability_contract": contract,
        "member_count": len(sealed),
        "state_counts": {
            state: sum(item["bundle_state"] == state for item in sealed)
            for state in BUNDLE_STATES
        },
        "members": sealed,
        "member_bindings_sha256": canonical_json_sha256(
            [_bundle_binding_projection(item) for item in sealed]
        ),
        "capability_matrix_sha256": canonical_json_sha256(
            [_capability_projection(item) for item in sealed]
        ),
        "software_authority": {
            "repository": "palette",
            "commit": _commit(palette_commit),
        },
        "created_at_utc": _utc_timestamp(created_at_utc, field="created_at_utc"),
        "safety": SAFETY,
    }
    result = {**body, "record_sha256": canonical_json_sha256(body)}
    validate_validated_behavior_bundle_set(result, membership=validated_membership)
    return result


def validate_validated_behavior_bundle_set(
    value: object,
    *,
    membership: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate a bundle set against its exact normalized membership."""

    bundle_set = _plain(_mapping(value, field="bundle set"))
    _strict_json_size(bundle_set, limit=MAX_BUNDLE_SET_BYTES, field="bundle set")
    persisted = bundle_set.pop("record_sha256", None)
    if _digest(persisted, field="record_sha256") != canonical_json_sha256(bundle_set):
        _fail("Bundle-set record digest is stale.")
    required = {
        "schema_id",
        "schema_version",
        "method_id",
        "status",
        "bundle_set_id",
        "membership",
        "bundle_root",
        "bundle_profile",
        "capability_contract",
        "member_count",
        "state_counts",
        "members",
        "member_bindings_sha256",
        "capability_matrix_sha256",
        "software_authority",
        "created_at_utc",
        "safety",
    }
    if set(bundle_set) != required:
        _fail("Bundle-set field set is inexact.")
    if (
        bundle_set.get("schema_id") != BUNDLE_SET_SCHEMA_ID
        or bundle_set.get("schema_version") != BUNDLE_SET_SCHEMA_VERSION
        or bundle_set.get("method_id") != BUNDLE_SET_METHOD_ID
        or bundle_set.get("status") != BUNDLE_SET_STATUS
        or bundle_set.get("safety") != SAFETY
    ):
        _fail("Bundle-set identity, method, status, or safety is invalid.")
    _safe_id(bundle_set.get("bundle_set_id"), field="bundle_set_id")
    validated_membership = validate_validated_behavior_cohort_membership(membership)
    membership_binding = _mapping(bundle_set.get("membership"), field="membership")
    if set(membership_binding) != {
        "path",
        "file_sha256",
        "record_sha256",
        "members_sha256",
        "member_count",
    }:
        _fail("Bundle-set membership binding is inexact.")
    membership_path = _canonical_absolute_path(
        membership_binding.get("path"),
        field="membership.path",
        forbid_selectors=True,
    )
    normalized_membership_binding = {
        "path": str(membership_path),
        "file_sha256": _digest(
            membership_binding.get("file_sha256"), field="membership.file_sha256"
        ),
        "record_sha256": _digest(
            membership_binding.get("record_sha256"), field="membership.record_sha256"
        ),
        "members_sha256": _digest(
            membership_binding.get("members_sha256"), field="membership.members_sha256"
        ),
        "member_count": _positive_int(
            membership_binding.get("member_count"), field="membership.member_count"
        ),
    }
    if (
        normalized_membership_binding["record_sha256"]
        != validated_membership["record_sha256"]
        or normalized_membership_binding["members_sha256"]
        != validated_membership["members_sha256"]
        or normalized_membership_binding["member_count"]
        != validated_membership["member_count"]
    ):
        _fail("Bundle set binds another membership generation.")
    bundle_root = _canonical_absolute_path(
        bundle_set.get("bundle_root"), field="bundle_root", forbid_selectors=True
    )
    profile = _mapping(bundle_set.get("bundle_profile"), field="bundle_profile")
    if not profile:
        _fail("bundle_profile must not be empty.")
    contract = _validate_capability_contract(bundle_set.get("capability_contract"))
    raw_members = _list(bundle_set.get("members"), field="members")
    if len(raw_members) != validated_membership["member_count"] or bundle_set.get(
        "member_count"
    ) != len(raw_members):
        _fail("Bundle-set member count is stale.")
    members = [
        _validate_bundle_set_member(
            raw,
            membership_member=validated_membership["members"][index],
            capability_contract=contract,
            bundle_root=bundle_root,
        )
        for index, raw in enumerate(raw_members)
    ]
    expected_state_counts = {
        state: sum(item["bundle_state"] == state for item in members)
        for state in BUNDLE_STATES
    }
    state_counts = _mapping(bundle_set.get("state_counts"), field="state_counts")
    if (
        set(state_counts) != set(BUNDLE_STATES)
        or _plain(state_counts) != expected_state_counts
    ):
        _fail("Bundle-set state counts are stale or incomplete.")
    if canonical_json_sha256(
        [_bundle_binding_projection(item) for item in members]
    ) != _digest(
        bundle_set.get("member_bindings_sha256"), field="member_bindings_sha256"
    ):
        _fail("Bundle-set aggregate binding digest is stale.")
    if canonical_json_sha256(
        [_capability_projection(item) for item in members]
    ) != _digest(
        bundle_set.get("capability_matrix_sha256"), field="capability_matrix_sha256"
    ):
        _fail("Bundle-set aggregate capability digest is stale.")
    _validate_software(bundle_set.get("software_authority"))
    _utc_timestamp(bundle_set.get("created_at_utc"), field="created_at_utc")
    return _freeze(
        {
            **bundle_set,
            "membership": normalized_membership_binding,
            "bundle_profile": _plain(profile),
            "capability_contract": contract,
            "members": members,
            "record_sha256": persisted,
        }
    )


def _read_json_object(path: str | Path, *, field: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{field} does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidatedBehaviorCohortError(
            f"Cannot read strict JSON object from {source}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"{field} must contain one JSON object.")
    return value


def read_validated_behavior_cohort_membership(
    path: str | Path,
) -> Mapping[str, Any]:
    """Read one normalized membership without selecting a source adapter."""

    return validate_validated_behavior_cohort_membership(
        _read_json_object(path, field="membership")
    )


def read_validated_behavior_bundle_set(
    path: str | Path,
    *,
    membership: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read one generic bundle set against an already selected membership."""

    return validate_validated_behavior_bundle_set(
        _read_json_object(path, field="bundle set"), membership=membership
    )


__all__ = [
    "BUNDLE_SET_METHOD_ID",
    "BUNDLE_SET_SCHEMA_ID",
    "BUNDLE_SET_SCHEMA_VERSION",
    "BUNDLE_SET_STATUS",
    "BUNDLE_STATES",
    "CAPABILITY_CONTRACT_SCHEMA_ID",
    "CAPABILITY_CONTRACT_SCHEMA_VERSION",
    "CAPABILITY_STATES",
    "MEMBERSHIP_METHOD_ID",
    "MEMBERSHIP_REASON_CODES",
    "MEMBERSHIP_SCHEMA_ID",
    "MEMBERSHIP_SCHEMA_VERSION",
    "MEMBERSHIP_STATES",
    "MEMBERSHIP_STATUS",
    "ORDERING_POLICY",
    "RELOCATION_POLICY",
    "SAFETY",
    "SELECTOR_PATH_POLICY",
    "ValidatedBehaviorCohortError",
    "build_capability_contract",
    "build_validated_behavior_bundle_set",
    "build_validated_behavior_cohort_membership",
    "policy_envelope",
    "read_validated_behavior_bundle_set",
    "read_validated_behavior_cohort_membership",
    "validate_validated_behavior_bundle_set",
    "validate_validated_behavior_cohort_membership",
]
