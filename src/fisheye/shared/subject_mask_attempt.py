"""Deterministic scientific identity and immutable attempt lineage for masks."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID, uuid4

from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_COMPLETE,
)

SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_ID = "palette.subject_mask.scientific_identity"
SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION = 1
SUBJECT_MASK_ATTEMPT_SCHEMA_ID = "palette.subject_mask.attempt"
SUBJECT_MASK_ATTEMPT_SCHEMA_VERSION = 1


def _optional_run_name(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or "/" in normalized:
        raise ValueError(f"{name} must be one nonempty run name when provided.")
    return normalized


def build_subject_mask_scientific_identity(
    *,
    stage_kind: str,
    model: Mapping[str, Any],
    crop: Mapping[str, Any],
    pixels: Mapping[str, Any],
    row_identity: Mapping[str, Any],
    inference_contract: Mapping[str, Any],
) -> dict[str, object]:
    """Bind the scientific inputs independently of runtime and storage layout."""

    stage = str(stage_kind).strip()
    if stage not in {
        "raw_subject_mask",
        "refined_subject_mask",
        "subject_mask_quality",
    }:
        raise ValueError(f"Unsupported subject-mask scientific stage {stage!r}.")
    payload = {
        "stage_kind": stage,
        "model": dict(model),
        "crop": dict(crop),
        "pixels": dict(pixels),
        "row_identity": dict(row_identity),
        "inference_contract": dict(inference_contract),
    }
    canonical_json_bytes(payload)
    return {
        "schema_id": SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def build_subject_mask_attempt(
    *,
    scientific_identity: Mapping[str, Any],
    run_path: str,
    attempt_id: str | None = None,
    retry_of_attempt_id: str | None = None,
    supersedes_run: str | None = None,
) -> dict[str, object]:
    """Create one execution identity without changing scientific identity."""

    errors = validate_subject_mask_scientific_identity(scientific_identity)
    if errors:
        raise ValueError(
            "Invalid subject-mask scientific identity: " + "; ".join(errors)
        )
    resolved_attempt = str(UUID(attempt_id)) if attempt_id else str(uuid4())
    resolved_retry = (
        str(UUID(str(retry_of_attempt_id))) if retry_of_attempt_id is not None else None
    )
    if resolved_retry == resolved_attempt:
        raise ValueError("retry_of_attempt_id cannot equal attempt_id.")
    normalized_path = str(run_path).strip().strip("/")
    if not normalized_path or "/" not in normalized_path:
        raise ValueError("run_path must contain a family and run name.")
    payload = {
        "attempt_id": resolved_attempt,
        "scientific_identity_digest": scientific_identity["digest"],
        "run_path": normalized_path,
        "retry_of_attempt_id": resolved_retry,
        "supersedes_run": _optional_run_name(
            supersedes_run,
            name="supersedes_run",
        ),
        "retry_policy": "new_immutable_run_name_same_scientific_identity",
        "supersedes_policy": "explicit_predecessor_only_no_implicit_latest",
    }
    return {
        "schema_id": SUBJECT_MASK_ATTEMPT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_ATTEMPT_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def validate_subject_mask_scientific_identity(
    value: Mapping[str, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    if set(value) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "digest",
        "payload",
    }:
        errors.append("scientific identity envelope fields are not exact")
    payload = value.get("payload")
    if (
        value.get("schema_id") != SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_ID
        or value.get("schema_version")
        != SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION
        or value.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("scientific identity envelope mismatch")
    if not isinstance(payload, Mapping) or set(payload) != {
        "stage_kind",
        "model",
        "crop",
        "pixels",
        "row_identity",
        "inference_contract",
    }:
        errors.append("scientific identity payload fields are not exact")
    else:
        try:
            if value.get("digest") != canonical_json_sha256(payload):
                errors.append("scientific identity digest mismatch")
        except (TypeError, ValueError) as exc:
            errors.append(f"scientific identity is not strict JSON: {exc}")
    return tuple(errors)


def validate_subject_mask_attempt(value: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if set(value) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("attempt envelope fields are not exact")
    payload = value.get("payload")
    if (
        value.get("schema_id") != SUBJECT_MASK_ATTEMPT_SCHEMA_ID
        or value.get("schema_version") != SUBJECT_MASK_ATTEMPT_SCHEMA_VERSION
        or value.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("attempt envelope mismatch")
    if not isinstance(payload, Mapping) or set(payload) != {
        "attempt_id",
        "scientific_identity_digest",
        "run_path",
        "retry_of_attempt_id",
        "supersedes_run",
        "retry_policy",
        "supersedes_policy",
    }:
        errors.append("attempt payload fields are not exact")
        return tuple(errors)
    try:
        UUID(str(payload.get("attempt_id")))
        if payload.get("retry_of_attempt_id") is not None:
            UUID(str(payload.get("retry_of_attempt_id")))
    except (TypeError, ValueError):
        errors.append("attempt UUID lineage is invalid")
    if value.get("payload_digest") != canonical_json_sha256(payload):
        errors.append("attempt payload digest mismatch")
    return tuple(errors)


def resolve_subject_mask_attempt_lineage(
    *,
    parent: Any,
    current_run_name: str,
    scientific_identity: Mapping[str, Any],
    attempt: Mapping[str, Any],
    retry_of_attempt_id: str | None,
    supersedes_run: str | None,
    scientific_identity_attr: str = "subject_mask_scientific_identity",
    attempt_attr: str = "subject_mask_attempt",
) -> dict[str, object]:
    """Bind retry/supersession claims to immutable terminal sibling runs."""

    scientific_errors = validate_subject_mask_scientific_identity(scientific_identity)
    attempt_errors = validate_subject_mask_attempt(attempt)
    if scientific_errors or attempt_errors:
        raise ValueError(
            "Invalid subject-mask attempt records: "
            f"science={scientific_errors!r}, attempt={attempt_errors!r}."
        )
    attempt_id = str(attempt["payload"]["attempt_id"])
    retry_matches: list[tuple[str, Any, Mapping[str, Any]]] = []
    for sibling_name in parent.keys():
        if sibling_name == current_run_name:
            continue
        sibling = parent[sibling_name]
        sibling_attempt = sibling.attrs.get(attempt_attr)
        if not isinstance(sibling_attempt, Mapping):
            continue
        errors = validate_subject_mask_attempt(sibling_attempt)
        if errors:
            raise ValueError(
                f"Sibling {sibling_name!r} has malformed subject-mask attempt "
                f"metadata: {errors!r}."
            )
        sibling_attempt_id = str(sibling_attempt["payload"]["attempt_id"])
        if sibling_attempt_id == attempt_id:
            raise ValueError(
                f"Subject-mask attempt_id {attempt_id!r} is already in use by "
                f"{sibling_name!r}."
            )
        if retry_of_attempt_id is not None and sibling_attempt_id == str(
            retry_of_attempt_id
        ):
            retry_matches.append((str(sibling_name), sibling, sibling_attempt))

    retry_evidence: dict[str, object] | None = None
    if retry_of_attempt_id is not None:
        if len(retry_matches) != 1:
            raise ValueError(
                "retry_of_attempt_id must identify exactly one sibling attempt; "
                f"found {len(retry_matches)}."
            )
        retry_name, retry_group, retry_attempt = retry_matches[0]
        retry_science = retry_group.attrs.get(scientific_identity_attr)
        if (
            retry_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != "failed"
            or not isinstance(retry_science, Mapping)
            or retry_science.get("digest") != scientific_identity.get("digest")
        ):
            raise ValueError(
                "A retry must reference one failed attempt with the exact same "
                "scientific identity."
            )
        retry_evidence = {
            "run_name": retry_name,
            "run_path": (
                f"{str(getattr(parent, 'path', '')).strip('/')}/{retry_name}"
            ).strip("/"),
            "attempt_id": str(retry_of_attempt_id),
            "attempt_payload_digest": retry_attempt.get("payload_digest"),
            "scientific_identity_digest": retry_science.get("digest"),
            "completion_status": "failed",
        }

    supersedes_evidence: dict[str, object] | None = None
    if supersedes_run is not None:
        predecessor_name = str(supersedes_run).strip()
        if predecessor_name == current_run_name or predecessor_name not in parent:
            raise ValueError(
                "supersedes_run must identify a different existing sibling run."
            )
        predecessor = parent[predecessor_name]
        if predecessor.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
            raise ValueError("A superseded subject-mask run must be complete.")
        predecessor_attempt = predecessor.attrs.get(attempt_attr)
        predecessor_science = predecessor.attrs.get(scientific_identity_attr)
        supersedes_evidence = {
            "run_name": predecessor_name,
            "run_path": (
                f"{str(getattr(parent, 'path', '')).strip('/')}/{predecessor_name}"
            ).strip("/"),
            "completion_status": RUN_STATUS_COMPLETE,
            "attempt_payload_digest": (
                predecessor_attempt.get("payload_digest")
                if isinstance(predecessor_attempt, Mapping)
                else None
            ),
            "scientific_identity_digest": (
                predecessor_science.get("digest")
                if isinstance(predecessor_science, Mapping)
                else None
            ),
        }
    return {
        "retry_of": retry_evidence,
        "supersedes": supersedes_evidence,
        "lineage_policy": "explicit_terminal_sibling_binding_v1",
    }


__all__ = [
    "SUBJECT_MASK_ATTEMPT_SCHEMA_ID",
    "SUBJECT_MASK_ATTEMPT_SCHEMA_VERSION",
    "SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_ID",
    "SUBJECT_MASK_SCIENTIFIC_IDENTITY_SCHEMA_VERSION",
    "build_subject_mask_attempt",
    "build_subject_mask_scientific_identity",
    "resolve_subject_mask_attempt_lineage",
    "validate_subject_mask_attempt",
    "validate_subject_mask_scientific_identity",
]
