"""Persisted manifest and selection envelopes for refined-detection v1.

This module defines JSON-safe documents only.  It performs no Zarr I/O and
does not mutate selectors.  Future publishers must store the run-manifest
envelope at the exact run-group attribute declared below and validate it before
making a snapshot visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping
from uuid import UUID

import numpy as np

from fisheye.shared.instance_keys import (
    INSTANCE_KEY_ALGORITHM,
    INSTANCE_KEY_BBOX_QUANTIZATION,
    INSTANCE_KEY_CONTEXT_MANUAL_CURATION_ROW_ID,
    mint_manual_curation_instance_keys,
)
from fisheye.shared.zarr.refined_detection_schema import (
    REFINED_DETECTION_SCHEMA_V1,
    SOURCE_KIND_CODE_MAP,
    RefinedDetectionClippedBinding,
    RefinedDetectionDimensions,
    RefinedDetectionLineageProfile,
)
from fisheye.shared.zarr.refined_detection_storage import (
    RefinedDetectionStoragePlanSet,
)


REFINED_DETECTION_RUN_MANIFEST_SCHEMA_ID = (
    "palette.refined_detection.run_manifest"
)
REFINED_DETECTION_RUN_MANIFEST_SCHEMA_VERSION = 1
REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
REFINED_DETECTION_RUN_MANIFEST_PERSISTED_PATH = (
    "refined_detect_runs/<run>/zarr.json.attributes.run_manifest"
)

REFINED_DETECTION_AUTHORITY_SCHEMA_ID = (
    "palette.refined_detection.authoritative_selection"
)
REFINED_DETECTION_AUTHORITY_SCHEMA_VERSION = 1
REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE = "authoritative_run"
REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE = (
    "authoritative_run_provenance"
)

CANONICAL_JSON_DIGEST_ALGORITHM = "sha256_canonical_json_v1"
METADATA_DECLARATIONS_DIGEST_SCOPE = (
    "normalized_group_and_array_declarations_excluding_attributes"
)
REFINED_ROW_ID_ALLOCATOR_SCHEME = "monotonic_int64_nonreuse_v1"
MANUAL_INSTANCE_KEY_ALLOCATOR_SCHEME = (
    "blake2b64_manual_namespace_by_refined_row_id_v1"
)

REFINED_DETECTION_INTENDED_USES = (
    "analysis",
    "training",
    "analysis_and_training",
)
REFINED_DETECTION_RAW_FALLBACK_POLICIES = (
    "forbid",
    "allow_only_when_no_refined_authority",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def canonical_json_bytes(value: object) -> bytes:
    """Return strict deterministic UTF-8 JSON for contract digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(
            f"{name} must be a lowercase 64-character SHA-256 hex digest."
        )
    return normalized


def _require_text(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    return normalized


def _require_run_id(value: str, *, name: str = "run_id") -> str:
    normalized = _require_text(value, name=name)
    if "/" in normalized:
        raise ValueError(f"{name} must be a path-safe child-group name.")
    return normalized


def _require_uuid(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    try:
        parsed = UUID(normalized)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a canonical UUID string.") from exc
    canonical = str(parsed)
    if normalized != canonical:
        raise ValueError(f"{name} must use canonical lowercase UUID form.")
    return canonical


def _require_utc_timestamp(value: str, *, name: str) -> str:
    normalized = _require_text(value, name=name)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset.")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be expressed in UTC.")
    return normalized


def _array_values(value: Any, *, dtype: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=dtype)
    try:
        return np.asarray(value[...], dtype=dtype)
    except (IndexError, KeyError, TypeError):
        return np.asarray(value, dtype=dtype)


@dataclass(frozen=True)
class RefinedDetectionSnapshotLineage:
    """Cross-snapshot identity and monotonic allocation state."""

    lineage_id: str
    snapshot_id: str
    recording_identity: str
    next_refined_row_id: int
    parent_run_id: str | None = None
    parent_manifest_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lineage_id",
            _require_uuid(self.lineage_id, name="lineage_id"),
        )
        object.__setattr__(
            self,
            "snapshot_id",
            _require_uuid(self.snapshot_id, name="snapshot_id"),
        )
        object.__setattr__(
            self,
            "recording_identity",
            _require_text(self.recording_identity, name="recording_identity"),
        )
        if type(self.next_refined_row_id) is not int or self.next_refined_row_id < 0:
            raise ValueError("next_refined_row_id must be a nonnegative exact integer.")
        if (self.parent_run_id is None) != (self.parent_manifest_digest is None):
            raise ValueError(
                "parent_run_id and parent_manifest_digest must both be present or absent."
            )
        if self.parent_run_id is not None:
            object.__setattr__(
                self,
                "parent_run_id",
                _require_run_id(self.parent_run_id, name="parent_run_id"),
            )
            object.__setattr__(
                self,
                "parent_manifest_digest",
                _require_sha256(
                    str(self.parent_manifest_digest),
                    name="parent_manifest_digest",
                ),
            )

    def as_manifest(self) -> dict[str, object]:
        parent = (
            None
            if self.parent_run_id is None
            else {
                "run_id": self.parent_run_id,
                "run_manifest_digest": self.parent_manifest_digest,
            }
        )
        return {
            "lineage_id": self.lineage_id,
            "snapshot_id": self.snapshot_id,
            "parent_snapshot": parent,
            "refined_row_id_allocator": {
                "scheme": REFINED_ROW_ID_ALLOCATOR_SCHEME,
                "next_id": self.next_refined_row_id,
                "retired_ids": "never_reused",
            },
            "manual_instance_key_allocator": {
                "scheme": MANUAL_INSTANCE_KEY_ALLOCATOR_SCHEME,
                "algorithm": INSTANCE_KEY_ALGORITHM,
                "namespace": INSTANCE_KEY_CONTEXT_MANUAL_CURATION_ROW_ID,
                "recording_identity": self.recording_identity,
                "bbox_quantization": INSTANCE_KEY_BBOX_QUANTIZATION,
                "allocation_anchor": "refined_row_id",
                "refined_row_id_next": self.next_refined_row_id,
                "collision_policy": "reject_against_refined_and_source_keys",
            },
        }


@dataclass(frozen=True)
class RefinedDetectionSourceIdentity:
    """Exact immutable canonical raw-detection source binding."""

    run_id: str
    run_manifest_digest: str
    logical_content_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_run_id(self.run_id))
        object.__setattr__(
            self,
            "run_manifest_digest",
            _require_sha256(
                self.run_manifest_digest,
                name="source run_manifest_digest",
            ),
        )
        object.__setattr__(
            self,
            "logical_content_digest",
            _require_sha256(
                self.logical_content_digest,
                name="source logical_content_digest",
            ),
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "stage": "detect",
            "run_id": self.run_id,
            "logical_schema": {
                "id": "palette.stage.canonical_detection",
                "version": 1,
            },
            "run_manifest_digest": self.run_manifest_digest,
            "logical_content_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "logical_content_digest": self.logical_content_digest,
        }


def _reason_registry(
    values: Mapping[int | str, str],
    *,
    registry_id: str,
) -> dict[str, object]:
    normalized: dict[str, str] = {}
    for raw_code, raw_label in values.items():
        if isinstance(raw_code, bool):
            raise TypeError("Reason codes cannot be booleans.")
        try:
            code = int(raw_code)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid reason code {raw_code!r}.") from exc
        if str(raw_code).strip() not in {str(code), f"+{code}"}:
            raise ValueError(f"Reason code {raw_code!r} is not canonical decimal.")
        if not 0 <= code <= int(np.iinfo(np.uint16).max):
            raise ValueError("Reason codes must fit the canonical uint16 dtype.")
        label = str(raw_label).strip()
        if not _REASON_LABEL_RE.fullmatch(label):
            raise ValueError(
                "Reason labels must be lowercase snake-case identifiers."
            )
        key = str(code)
        if key in normalized:
            raise ValueError(f"Duplicate reason code {code}.")
        normalized[key] = label
    ordered = {
        key: normalized[key]
        for key in sorted(normalized, key=lambda item: int(item))
    }
    if ordered.get("0") != "none":
        raise ValueError("Each reason registry must define exact code 0 as 'none'.")
    registry_payload = {
        "schema_id": "palette.refined_detection.reason_registry",
        "schema_version": 1,
        "registry_id": registry_id,
        "codes": ordered,
    }
    return {
        **registry_payload,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "digest": canonical_json_sha256(registry_payload),
    }


def build_refined_detection_run_manifest(
    *,
    run_id: str,
    dimensions: RefinedDetectionDimensions,
    storage_plan: RefinedDetectionStoragePlanSet,
    lineage: RefinedDetectionSnapshotLineage,
    source: RefinedDetectionSourceIdentity,
    instance_reason_codes: Mapping[int | str, str],
    source_reason_codes: Mapping[int | str, str],
    metadata_declarations_digest: str,
    selector_eligible: bool,
    clipped_binding: RefinedDetectionClippedBinding | None = None,
) -> dict[str, object]:
    """Build the exact run-group ``run_manifest`` attribute envelope."""

    resolved_run_id = _require_run_id(run_id)
    if type(selector_eligible) is not bool:
        raise TypeError("selector_eligible must be an exact boolean.")
    if storage_plan.dimensions != dimensions:
        raise ValueError("Storage-plan dimensions must equal logical dimensions.")
    if (
        dimensions.lineage_profile
        is RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT
    ) != (clipped_binding is not None):
        raise ValueError("clipped_binding must match the logical lineage profile.")
    metadata_digest = _require_sha256(
        metadata_declarations_digest,
        name="metadata_declarations_digest",
    )
    logical_schema = REFINED_DETECTION_SCHEMA_V1.as_manifest(
        dimensions=dimensions,
        clipped_binding=clipped_binding,
    )
    payload: dict[str, object] = {
        "run_id": resolved_run_id,
        "stage": "refined_detect",
        "publication": {
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": selector_eligible,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": (
                METADATA_DECLARATIONS_DIGEST_SCOPE
            ),
            "metadata_declarations_digest_algorithm": (
                CANONICAL_JSON_DIGEST_ALGORITHM
            ),
            "metadata_declarations_digest": metadata_digest,
        },
        "logical_schema": logical_schema,
        "storage_plan": storage_plan.as_manifest(),
        "snapshot_lineage": lineage.as_manifest(),
        "source_detection": source.as_manifest(),
        "reason_registries": {
            "instances": _reason_registry(
                instance_reason_codes,
                registry_id="instances.reason_codes.v1",
            ),
            "source_detections": _reason_registry(
                source_reason_codes,
                registry_id="source_detections.reason_codes.v1",
            ),
        },
    }
    envelope = {
        "schema_id": REFINED_DETECTION_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_RUN_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def validate_refined_detection_run_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the immutable envelope without opening any array payload."""

    errors: list[str] = []
    if set(manifest) != {
        "schema_id",
        "schema_version",
        "persisted_attribute",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("run manifest envelope has an unexpected field set")
    if manifest.get("schema_id") != REFINED_DETECTION_RUN_MANIFEST_SCHEMA_ID:
        errors.append("run manifest schema_id mismatch")
    if manifest.get("schema_version") != REFINED_DETECTION_RUN_MANIFEST_SCHEMA_VERSION:
        errors.append("run manifest schema_version mismatch")
    if manifest.get("persisted_attribute") != REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE:
        errors.append("run manifest persisted_attribute mismatch")
    if manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("run manifest digest_algorithm mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "run manifest payload must be an object")
    try:
        expected_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        return (*errors, f"run manifest payload is not strict JSON: {exc}")
    if manifest.get("payload_digest") != expected_digest:
        errors.append("run manifest payload_digest mismatch")
    if payload.get("stage") != "refined_detect":
        errors.append("run manifest stage mismatch")
    if set(payload) != {
        "run_id",
        "stage",
        "publication",
        "logical_schema",
        "storage_plan",
        "snapshot_lineage",
        "source_detection",
        "reason_registries",
    }:
        errors.append("run manifest payload has an unexpected field set")
    try:
        _require_run_id(str(payload.get("run_id") or ""))
    except ValueError as exc:
        errors.append(str(exc))
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("run manifest publication must be an object")
    else:
        if publication.get("completion_status") != "complete":
            errors.append("run manifest completion_status must be complete")
        if type(publication.get("stage_selector_eligible")) is not bool:
            errors.append("stage_selector_eligible must be a JSON boolean")
        if publication.get("metadata_state") != "direct_and_consolidated_validated":
            errors.append("metadata_state must prove direct/consolidated validation")
        if publication.get("completion_contract") != "palette.zarr_run_completion.v1":
            errors.append("completion_contract mismatch")
        if publication.get("metadata_declarations_digest_scope") != (
            METADATA_DECLARATIONS_DIGEST_SCOPE
        ):
            errors.append("metadata_declarations_digest_scope mismatch")
        if publication.get("metadata_declarations_digest_algorithm") != (
            CANONICAL_JSON_DIGEST_ALGORITHM
        ):
            errors.append("metadata_declarations_digest_algorithm mismatch")
        try:
            _require_sha256(
                str(publication.get("metadata_declarations_digest") or ""),
                name="metadata_declarations_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))

    logical = payload.get("logical_schema")
    if not isinstance(logical, Mapping):
        errors.append("logical_schema must be an object")
    else:
        if logical.get("schema_id") != REFINED_DETECTION_SCHEMA_V1.schema_id:
            errors.append("logical_schema schema_id mismatch")
        if logical.get("schema_version") != REFINED_DETECTION_SCHEMA_V1.schema_version:
            errors.append("logical_schema schema_version mismatch")
        dimensions = logical.get("dimensions")
        if (
            not isinstance(dimensions, Mapping)
            or type(dimensions.get("n_frames")) is not int
            or int(dimensions.get("n_frames", 0)) <= 0
        ):
            errors.append("logical_schema requires a positive n_frames")
        profile = (
            dimensions.get("lineage_profile")
            if isinstance(dimensions, Mapping)
            else None
        )
        clipped = logical.get("clipped_binding")
        expected_bindings = (
            (
                *REFINED_DETECTION_SCHEMA_V1.core_bindings,
                *REFINED_DETECTION_SCHEMA_V1.clipped_lineage_bindings,
            )
            if profile
            == RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT.value
            else REFINED_DETECTION_SCHEMA_V1.core_bindings
        )
        if logical.get("bindings") != [
            binding.as_manifest() for binding in expected_bindings
        ]:
            errors.append("logical_schema binding declarations mismatch")
        if logical.get("array_contracts") != (
            REFINED_DETECTION_SCHEMA_V1.contracts.as_manifest()
        ):
            errors.append("logical_schema array contract catalog mismatch")
        if profile == RefinedDetectionLineageProfile.CLIPPED_RECORDING_SNAPSHOT.value:
            if not isinstance(clipped, Mapping):
                errors.append("clipped lineage profile requires clipped_binding")
            elif (
                clipped.get("schema_id")
                != "palette.refined_detection.clipped_binding"
                or clipped.get("schema_version") != 1
                or clipped.get("camera_cardinality") != 1
                or clipped.get("clip_ordinal_scope")
                != "snapshot_global_within_single_camera"
            ):
                errors.append("clipped_binding schema or scope mismatch")
        elif clipped is not None:
            errors.append("full-acquisition logical schema cannot carry clipped_binding")

    storage = payload.get("storage_plan")
    if not isinstance(storage, Mapping):
        errors.append("storage_plan must be an object")
    else:
        if storage.get("schema_id") != "palette.stage_storage.refined_detection":
            errors.append("storage_plan schema_id mismatch")
        if storage.get("schema_version") != 1:
            errors.append("storage_plan schema_version mismatch")
        if storage.get("logical_stage_schema") != {
            "id": REFINED_DETECTION_SCHEMA_V1.schema_id,
            "version": REFINED_DETECTION_SCHEMA_V1.schema_version,
        }:
            errors.append("storage_plan logical schema binding mismatch")
        if isinstance(logical, Mapping) and storage.get("dimensions") != (
            logical.get("dimensions")
        ):
            errors.append("storage_plan dimensions differ from logical schema")
        storage_arrays = storage.get("arrays")
        logical_bindings = logical.get("bindings")
        binding_paths = (
            [binding.get("path") for binding in logical_bindings]
            if isinstance(logical_bindings, list)
            and all(isinstance(binding, Mapping) for binding in logical_bindings)
            else []
        )
        storage_paths = (
            [entry.get("path") for entry in storage_arrays]
            if isinstance(storage_arrays, list)
            and all(isinstance(entry, Mapping) for entry in storage_arrays)
            else []
        )
        if storage_paths != binding_paths:
            errors.append("storage_plan array paths differ from logical bindings")
        if storage.get("profile_status") != (
            "resolved_plan_evidence_not_a_production_default_promotion"
        ):
            errors.append("storage_plan profile_status mismatch")

    lineage = payload.get("snapshot_lineage")
    if not isinstance(lineage, Mapping):
        errors.append("snapshot_lineage must be an object")
    else:
        for name in ("lineage_id", "snapshot_id"):
            try:
                _require_uuid(str(lineage.get(name) or ""), name=name)
            except ValueError as exc:
                errors.append(str(exc))
        parent = lineage.get("parent_snapshot")
        if parent is not None:
            if not isinstance(parent, Mapping):
                errors.append("parent_snapshot must be null or an object")
            else:
                try:
                    _require_run_id(
                        str(parent.get("run_id") or ""),
                        name="parent run_id",
                    )
                    _require_sha256(
                        str(parent.get("run_manifest_digest") or ""),
                        name="parent run_manifest_digest",
                    )
                except ValueError as exc:
                    errors.append(str(exc))
        row_allocator = lineage.get("refined_row_id_allocator")
        if not isinstance(row_allocator, Mapping):
            errors.append("refined_row_id_allocator must be an object")
        else:
            if row_allocator.get("scheme") != REFINED_ROW_ID_ALLOCATOR_SCHEME:
                errors.append("refined_row_id allocator scheme mismatch")
            if type(row_allocator.get("next_id")) is not int or int(
                row_allocator.get("next_id", -1)
            ) < 0:
                errors.append("refined_row_id allocator next_id is invalid")
        key_allocator = lineage.get("manual_instance_key_allocator")
        if not isinstance(key_allocator, Mapping):
            errors.append("manual_instance_key_allocator must be an object")
        else:
            expected_key_fields = {
                "scheme": MANUAL_INSTANCE_KEY_ALLOCATOR_SCHEME,
                "algorithm": INSTANCE_KEY_ALGORITHM,
                "namespace": INSTANCE_KEY_CONTEXT_MANUAL_CURATION_ROW_ID,
                "bbox_quantization": INSTANCE_KEY_BBOX_QUANTIZATION,
                "allocation_anchor": "refined_row_id",
                "collision_policy": "reject_against_refined_and_source_keys",
            }
            for name, expected in expected_key_fields.items():
                if key_allocator.get(name) != expected:
                    errors.append(f"manual instance-key allocator {name} mismatch")
            if not str(key_allocator.get("recording_identity") or "").strip():
                errors.append("manual instance-key recording_identity is empty")
            if isinstance(row_allocator, Mapping) and key_allocator.get(
                "refined_row_id_next"
            ) != row_allocator.get("next_id"):
                errors.append("manual instance-key allocator high-water mark mismatch")

    source = payload.get("source_detection")
    if not isinstance(source, Mapping):
        errors.append("source_detection must be an object")
    else:
        if source.get("stage") != "detect" or source.get("logical_schema") != {
            "id": "palette.stage.canonical_detection",
            "version": 1,
        }:
            errors.append("source_detection schema binding mismatch")
        try:
            _require_run_id(str(source.get("run_id") or ""), name="source run_id")
            _require_sha256(
                str(source.get("run_manifest_digest") or ""),
                name="source run_manifest_digest",
            )
            _require_sha256(
                str(source.get("logical_content_digest") or ""),
                name="source logical_content_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))

    reason_registries = payload.get("reason_registries")
    if not isinstance(reason_registries, Mapping) or set(reason_registries) != {
        "instances",
        "source_detections",
    }:
        errors.append("reason_registries must contain exact instances/source maps")
    else:
        for name, expected_id in (
            ("instances", "instances.reason_codes.v1"),
            ("source_detections", "source_detections.reason_codes.v1"),
        ):
            registry = reason_registries.get(name)
            if not isinstance(registry, Mapping):
                errors.append(f"{name} reason registry must be an object")
                continue
            registry_payload = {
                "schema_id": registry.get("schema_id"),
                "schema_version": registry.get("schema_version"),
                "registry_id": registry.get("registry_id"),
                "codes": registry.get("codes"),
            }
            if registry_payload["schema_id"] != (
                "palette.refined_detection.reason_registry"
            ) or registry_payload["schema_version"] != 1:
                errors.append(f"{name} reason registry schema mismatch")
            if registry_payload["registry_id"] != expected_id:
                errors.append(f"{name} reason registry_id mismatch")
            codes = registry_payload["codes"]
            if not isinstance(codes, Mapping) or codes.get("0") != "none":
                errors.append(f"{name} reason registry must define code 0 as none")
            if registry.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
                errors.append(f"{name} reason registry digest algorithm mismatch")
            if registry.get("digest") != canonical_json_sha256(registry_payload):
                errors.append(f"{name} reason registry digest mismatch")
    return tuple(errors)


def validate_refined_detection_authority_provenance(
    authority: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the parent authority envelope independently of Zarr I/O."""

    errors: list[str] = []
    if set(authority) != {
        "schema_id",
        "schema_version",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("authority envelope has an unexpected field set")
    if authority.get("schema_id") != REFINED_DETECTION_AUTHORITY_SCHEMA_ID:
        errors.append("authority schema_id mismatch")
    if authority.get("schema_version") != REFINED_DETECTION_AUTHORITY_SCHEMA_VERSION:
        errors.append("authority schema_version mismatch")
    if authority.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("authority digest_algorithm mismatch")
    payload = authority.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "authority payload must be an object")
    if authority.get("payload_digest") != canonical_json_sha256(payload):
        errors.append("authority payload_digest mismatch")
    if set(payload) != {
        "run_id",
        "run_manifest_digest",
        "review_state",
        "review_method",
        "intended_use",
        "approved_by",
        "approved_at_utc",
        "git_sha",
        "note",
    }:
        errors.append("authority payload has an unexpected field set")
    try:
        _require_run_id(str(payload.get("run_id") or ""))
        _require_sha256(
            str(payload.get("run_manifest_digest") or ""),
            name="run_manifest_digest",
        )
        _require_text(str(payload.get("review_method") or ""), name="review_method")
        _require_text(str(payload.get("approved_by") or ""), name="approved_by")
        _require_utc_timestamp(
            str(payload.get("approved_at_utc") or ""),
            name="approved_at_utc",
        )
    except ValueError as exc:
        errors.append(str(exc))
    if payload.get("review_state") != "approved":
        errors.append("authority review_state must be approved")
    if payload.get("intended_use") not in REFINED_DETECTION_INTENDED_USES:
        errors.append("authority intended_use is not in the frozen enum")
    if payload.get("git_sha") is not None and not str(payload.get("git_sha")).strip():
        errors.append("authority git_sha must be null or nonempty")
    return tuple(errors)


def validate_refined_detection_snapshot_identity(
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
    parent_manifest: Mapping[str, Any] | None = None,
    parent_arrays: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Validate manifest allocation state and an optional parent transition."""

    errors = list(validate_refined_detection_run_manifest(manifest))
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return tuple(errors)
    lineage = payload.get("snapshot_lineage")
    if not isinstance(lineage, Mapping):
        return (*errors, "snapshot_lineage must be an object")
    allocator = lineage.get("refined_row_id_allocator")
    if not isinstance(allocator, Mapping):
        return (*errors, "refined_row_id_allocator must be an object")
    next_id = allocator.get("next_id")
    if type(next_id) is not int or next_id < 0:
        return (*errors, "refined_row_id next_id must be nonnegative")
    required_paths = (
        "instances/refined_row_ids",
        "instances/instance_key",
        "instances/source_kind_codes",
        "instances/frame_indices",
        "instances/bbox_norm_coords",
        "instances/class_ids",
    )
    missing = [path for path in required_paths if path not in arrays]
    if missing:
        return (*errors, f"identity validation is missing arrays: {missing!r}")
    row_ids = _array_values(
        arrays["instances/refined_row_ids"],
        dtype=np.int64,
    )
    keys = _array_values(arrays["instances/instance_key"], dtype=np.uint64)
    if row_ids.size and int(np.max(row_ids)) >= next_id:
        errors.append("refined_row_ids must be below allocator next_id")

    kind = _array_values(
        arrays["instances/source_kind_codes"],
        dtype=np.uint8,
    )
    manual = kind == SOURCE_KIND_CODE_MAP["manual"]
    manual_allocator = lineage.get("manual_instance_key_allocator")

    def validate_manual_keys(mask: np.ndarray) -> None:
        if not isinstance(manual_allocator, Mapping) or not bool(np.any(mask)):
            return
        expected = mint_manual_curation_instance_keys(
            recording_identity=str(manual_allocator.get("recording_identity") or ""),
            refined_row_ids=row_ids[mask],
            frame_indices=_array_values(
                arrays["instances/frame_indices"],
                dtype=np.int64,
            )[mask],
            bbox_norm_coords=_array_values(
                arrays["instances/bbox_norm_coords"],
                dtype=np.float32,
            )[mask],
            class_ids=_array_values(
                arrays["instances/class_ids"],
                dtype=np.int32,
            )[mask],
        )
        if not np.array_equal(keys[mask], expected):
            errors.append("manual instance_key values do not match the frozen allocator")

    if parent_manifest is None and parent_arrays is None:
        if lineage.get("parent_snapshot") is not None:
            errors.append("parent manifest/arrays are required for a successor snapshot")
        validate_manual_keys(manual)
        return tuple(errors)
    if parent_manifest is None or parent_arrays is None:
        return (*errors, "parent manifest and parent arrays must be provided together")
    parent_errors = validate_refined_detection_run_manifest(parent_manifest)
    errors.extend(f"parent: {error}" for error in parent_errors)
    parent_payload = parent_manifest.get("payload")
    if not isinstance(parent_payload, Mapping):
        return tuple(errors)
    parent_lineage = parent_payload.get("snapshot_lineage")
    if not isinstance(parent_lineage, Mapping):
        return (*errors, "parent snapshot_lineage must be an object")
    parent_ref = lineage.get("parent_snapshot")
    expected_parent_ref = {
        "run_id": parent_payload.get("run_id"),
        "run_manifest_digest": parent_manifest.get("payload_digest"),
    }
    if parent_ref != expected_parent_ref:
        errors.append("parent_snapshot does not bind the supplied parent manifest")
    if lineage.get("lineage_id") != parent_lineage.get("lineage_id"):
        errors.append("successor lineage_id differs from parent")
    parent_allocator = parent_lineage.get("refined_row_id_allocator")
    if not isinstance(parent_allocator, Mapping):
        return (*errors, "parent refined_row_id_allocator must be an object")
    parent_next = parent_allocator.get("next_id")
    if type(parent_next) is not int or parent_next < 0:
        return (*errors, "parent refined_row_id next_id must be nonnegative")
    if next_id < parent_next:
        errors.append("successor refined_row_id next_id regressed")
    validate_manual_keys(manual & (row_ids >= parent_next))

    parent_required_paths = (
        "instances/refined_row_ids",
        "instances/instance_key",
    )
    parent_missing = [
        path for path in parent_required_paths if path not in parent_arrays
    ]
    if parent_missing:
        return (
            *errors,
            f"parent identity validation is missing arrays: {parent_missing!r}",
        )
    parent_row_ids = _array_values(
        parent_arrays["instances/refined_row_ids"],
        dtype=np.int64,
    )
    parent_keys = _array_values(
        parent_arrays["instances/instance_key"],
        dtype=np.uint64,
    )
    parent_key_by_row = {
        int(row_id): int(key)
        for row_id, key in zip(
            parent_row_ids.tolist(),
            parent_keys.tolist(),
            strict=True,
        )
    }
    for row_id, key in zip(row_ids.tolist(), keys.tolist(), strict=True):
        if int(row_id) < parent_next:
            if int(row_id) not in parent_key_by_row:
                errors.append(
                    f"retired refined_row_id {int(row_id)} was reused by successor"
                )
            elif parent_key_by_row[int(row_id)] != int(key):
                errors.append(
                    f"surviving refined_row_id {int(row_id)} changed instance_key"
                )
    return tuple(errors)


def build_refined_detection_authority_provenance(
    *,
    run_id: str,
    run_manifest_digest: str,
    approved_by: str,
    approved_at_utc: str,
    review_method: str,
    intended_use: str,
    git_sha: str | None = None,
    note: str = "",
) -> dict[str, object]:
    """Build the exact parent ``authoritative_run_provenance`` envelope."""

    resolved_use = str(intended_use).strip()
    if resolved_use not in REFINED_DETECTION_INTENDED_USES:
        raise ValueError(
            f"intended_use must be one of {REFINED_DETECTION_INTENDED_USES!r}."
        )
    payload = {
        "run_id": _require_run_id(run_id),
        "run_manifest_digest": _require_sha256(
            run_manifest_digest,
            name="run_manifest_digest",
        ),
        "review_state": "approved",
        "review_method": _require_text(review_method, name="review_method"),
        "intended_use": resolved_use,
        "approved_by": _require_text(approved_by, name="approved_by"),
        "approved_at_utc": _require_utc_timestamp(
            approved_at_utc,
            name="approved_at_utc",
        ),
        "git_sha": None if git_sha is None else _require_text(git_sha, name="git_sha"),
        "note": str(note),
    }
    return {
        "schema_id": REFINED_DETECTION_AUTHORITY_SCHEMA_ID,
        "schema_version": REFINED_DETECTION_AUTHORITY_SCHEMA_VERSION,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }


def refined_detection_selection_contract_manifest() -> dict[str, object]:
    """Return the frozen production selection order for Palette and Crimson."""

    return {
        "schema_id": "palette.refined_detection.selection_contract",
        "schema_version": 1,
        "request": {
            "fields": {
                "stage": ["refined_detect", "detect"],
                "run": "required_for_explicit_selection",
                "raw_fallback_policy": list(
                    REFINED_DETECTION_RAW_FALLBACK_POLICIES
                ),
            },
            "default_raw_fallback_policy": "forbid",
        },
        "order": [
            "explicit_refined_v1",
            "approved_authoritative_refined_v1",
            "explicitly_permitted_canonical_raw",
        ],
        "explicit_refined_v1": {
            "requirements": [
                "run_exists",
                "run_manifest_valid",
                "completion_status_complete",
                "stage_selector_eligible_true",
                "direct_consolidated_equivalence_valid",
            ],
            "approval_required": False,
            "failure": "terminal_error_never_raw_fallback",
        },
        "approved_authoritative_refined_v1": {
            "pointer_path": (
                "refined_detect_runs/zarr.json.attributes.authoritative_run"
            ),
            "provenance_path": (
                "refined_detect_runs/zarr.json.attributes."
                "authoritative_run_provenance"
            ),
            "requirements": [
                "pointer_and_provenance_run_match",
                "provenance_manifest_digest_matches_run_manifest",
                "review_state_approved",
                "intended_use_is_frozen_enum",
                "completion_status_complete",
                "stage_selector_eligible_true",
            ],
            "invalid_pointer": "terminal_error_never_raw_fallback",
        },
        "canonical_raw_fallback": {
            "allowed_only_when": [
                "no_explicit_refined_request",
                "no_authoritative_refined_pointer",
                "raw_fallback_policy_allow_only_when_no_refined_authority",
            ],
            "invalid_refined_is_absence": False,
        },
        "benchmark_exception": (
            "selector_ineligible_direct paths require an explicit benchmark-only "
            "API outside production selection"
        ),
        "legacy": "separate_adapter_no_v1_alias_or_dtype_probe",
    }


__all__ = [
    "CANONICAL_JSON_DIGEST_ALGORITHM",
    "MANUAL_INSTANCE_KEY_ALLOCATOR_SCHEME",
    "METADATA_DECLARATIONS_DIGEST_SCOPE",
    "REFINED_DETECTION_AUTHORITY_PROVENANCE_ATTRIBUTE",
    "REFINED_DETECTION_AUTHORITY_RUN_ATTRIBUTE",
    "REFINED_DETECTION_AUTHORITY_SCHEMA_ID",
    "REFINED_DETECTION_AUTHORITY_SCHEMA_VERSION",
    "REFINED_DETECTION_INTENDED_USES",
    "REFINED_DETECTION_RAW_FALLBACK_POLICIES",
    "REFINED_DETECTION_RUN_MANIFEST_ATTRIBUTE",
    "REFINED_DETECTION_RUN_MANIFEST_PERSISTED_PATH",
    "REFINED_DETECTION_RUN_MANIFEST_SCHEMA_ID",
    "REFINED_DETECTION_RUN_MANIFEST_SCHEMA_VERSION",
    "REFINED_ROW_ID_ALLOCATOR_SCHEME",
    "RefinedDetectionSnapshotLineage",
    "RefinedDetectionSourceIdentity",
    "build_refined_detection_authority_provenance",
    "build_refined_detection_run_manifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "refined_detection_selection_contract_manifest",
    "validate_refined_detection_run_manifest",
    "validate_refined_detection_snapshot_identity",
    "validate_refined_detection_authority_provenance",
]
