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
    RefinedDetectionClipBinding,
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
METADATA_DECLARATIONS_SCHEMA_ID = (
    "palette.refined_detection.metadata_declarations"
)
METADATA_DECLARATIONS_SCHEMA_VERSION = 1
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


def _metadata_without_consolidation(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(value)
    normalized.pop("consolidated_metadata", None)
    return normalized


def normalize_refined_detection_metadata_declarations(
    direct_metadata_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_metadata_by_path: Mapping[str, Mapping[str, Any]],
    dimensions: RefinedDetectionDimensions,
) -> dict[str, object]:
    """Normalize the exact direct Zarr-v3 declaration tree for one run.

    Paths are relative to ``refined_detect_runs/<run>`` and use ``""`` for
    that run-group root. ``consolidated_metadata_by_path`` is the matching
    subtree extracted from the archive root's inline consolidated metadata and
    rebased to the same path basis. Every direct/consolidated declaration is
    compared before attributes and consolidation envelopes are removed from
    the digest surface.
    """

    if not isinstance(direct_metadata_by_path, Mapping):
        raise TypeError("direct_metadata_by_path must be a declaration mapping.")
    if not isinstance(consolidated_metadata_by_path, Mapping):
        raise TypeError(
            "consolidated_metadata_by_path must be a declaration mapping."
        )
    array_paths = REFINED_DETECTION_SCHEMA_V1.binding_paths_for(dimensions)
    group_paths = ("", "instances", "source_detections")
    expected_paths = {*group_paths, *array_paths}
    direct_paths = set(direct_metadata_by_path)
    if direct_paths != expected_paths:
        raise ValueError(
            "Refined direct metadata declaration paths must be exact; "
            f"missing={sorted(expected_paths - direct_paths)!r}, "
            f"unexpected={sorted(direct_paths - expected_paths)!r}."
        )
    consolidated_paths = set(consolidated_metadata_by_path)
    if consolidated_paths != expected_paths:
        raise ValueError(
            "Refined consolidated metadata declaration paths must be exact; "
            f"missing={sorted(expected_paths - consolidated_paths)!r}, "
            f"unexpected={sorted(consolidated_paths - expected_paths)!r}."
        )
    direct: dict[str, Mapping[str, Any]] = {}
    for path in sorted(expected_paths):
        declaration = direct_metadata_by_path[path]
        if not isinstance(declaration, Mapping):
            raise TypeError(f"Zarr metadata declaration {path!r} must be an object.")
        canonical_json_bytes(declaration)
        direct[path] = declaration
    for path in sorted(expected_paths):
        candidate = consolidated_metadata_by_path[path]
        if not isinstance(candidate, Mapping):
            raise TypeError(f"Consolidated declaration {path!r} must be an object.")
        canonical_json_bytes(candidate)
        if _metadata_without_consolidation(candidate) != (
            _metadata_without_consolidation(direct[path])
        ):
            raise ValueError(
                f"Direct and consolidated metadata differ at {path!r}."
            )

    normalized: dict[str, dict[str, Any]] = {}
    for path in sorted(expected_paths):
        declaration = dict(direct[path])
        if declaration.get("zarr_format") != 3:
            raise ValueError(f"Zarr declaration {path!r} must use zarr_format 3.")
        attributes = declaration.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError(f"Zarr declaration {path!r} requires object attributes.")
        if path in group_paths:
            expected_fields = {"zarr_format", "node_type", "attributes"}
            if set(declaration) != expected_fields:
                raise ValueError(
                    f"Zarr group declaration {path!r} has an unexpected field set."
                )
            if declaration.get("node_type") != "group":
                raise ValueError(f"Zarr declaration {path!r} must be a group.")
        else:
            required_fields = {
                "zarr_format",
                "node_type",
                "shape",
                "data_type",
                "chunk_grid",
                "chunk_key_encoding",
                "fill_value",
                "codecs",
                "attributes",
                "storage_transformers",
            }
            optional_fields = {"dimension_names"}
            if not required_fields.issubset(declaration) or not set(
                declaration
            ).issubset(required_fields | optional_fields):
                raise ValueError(
                    f"Zarr array declaration {path!r} has an unexpected field set."
                )
            if declaration.get("node_type") != "array":
                raise ValueError(f"Zarr declaration {path!r} must be an array.")
            if not isinstance(declaration.get("shape"), list) or not all(
                type(value) is int and value >= 0
                for value in declaration["shape"]
            ):
                raise ValueError(f"Zarr array declaration {path!r} has invalid shape.")
            if not isinstance(declaration.get("chunk_grid"), Mapping):
                raise ValueError(f"Zarr array declaration {path!r} lacks chunk_grid.")
            if not isinstance(declaration.get("chunk_key_encoding"), Mapping):
                raise ValueError(
                    f"Zarr array declaration {path!r} lacks chunk_key_encoding."
                )
            if not isinstance(declaration.get("codecs"), list):
                raise ValueError(f"Zarr array declaration {path!r} lacks codec list.")
            if not isinstance(declaration.get("storage_transformers"), list):
                raise ValueError(
                    f"Zarr array declaration {path!r} lacks storage_transformers list."
                )
        declaration.pop("attributes")
        declaration.pop("consolidated_metadata", None)
        normalized[path] = declaration

    document: dict[str, object] = {
        "schema_id": METADATA_DECLARATIONS_SCHEMA_ID,
        "schema_version": METADATA_DECLARATIONS_SCHEMA_VERSION,
        "path_basis": "relative_to_refined_detect_run_group_empty_string_is_root",
        "included_nodes": "exact_run_root_groups_and_schema_arrays",
        "excluded_fields": ["attributes", "consolidated_metadata"],
        "declarations": normalized,
    }
    canonical_json_bytes(document)
    return document


def refined_detection_metadata_declarations_digest(
    direct_metadata_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_metadata_by_path: Mapping[str, Mapping[str, Any]],
    dimensions: RefinedDetectionDimensions,
) -> str:
    """Digest the executable normalized direct/consolidated declaration tree."""

    return canonical_json_sha256(
        normalize_refined_detection_metadata_declarations(
            direct_metadata_by_path,
            consolidated_metadata_by_path=consolidated_metadata_by_path,
            dimensions=dimensions,
        )
    )


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
        if str(raw_code).strip() != str(code):
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


def _validate_reason_registry_document(
    registry: Mapping[str, Any],
    *,
    name: str,
    expected_id: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    if set(registry) != {
        "schema_id",
        "schema_version",
        "registry_id",
        "codes",
        "digest_algorithm",
        "digest",
    }:
        errors.append(f"{name} reason registry has an unexpected field set")
    codes = registry.get("codes")
    if not isinstance(codes, Mapping):
        return (*errors, f"{name} reason registry codes must be an object")
    if any(type(key) is not str for key in codes) or any(
        type(label) is not str for label in codes.values()
    ):
        errors.append(f"{name} reason registry codes and labels must be JSON strings")
    try:
        expected = _reason_registry(codes, registry_id=expected_id)
    except (TypeError, ValueError) as exc:
        errors.append(f"{name} reason registry is invalid: {exc}")
        return tuple(errors)
    if dict(registry) != expected:
        errors.append(f"{name} reason registry is not in canonical persisted form")
    return tuple(errors)


def build_refined_detection_run_manifest(
    *,
    run_id: str,
    dimensions: RefinedDetectionDimensions,
    storage_plan: RefinedDetectionStoragePlanSet,
    lineage: RefinedDetectionSnapshotLineage,
    source: RefinedDetectionSourceIdentity,
    instance_reason_codes: Mapping[int | str, str],
    source_reason_codes: Mapping[int | str, str],
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
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
    metadata_digest = refined_detection_metadata_declarations_digest(
        direct_metadata_declarations,
        consolidated_metadata_by_path=consolidated_metadata_declarations,
        dimensions=dimensions,
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


def parse_refined_detection_clipped_binding(
    value: Mapping[str, Any],
) -> RefinedDetectionClippedBinding:
    """Parse and deeply validate the exact persisted clipped-binding document."""

    expected_fields = {
        "schema_id",
        "schema_version",
        "collection_id",
        "collection_manifest_digest",
        "camera_cardinality",
        "camera_serial",
        "clip_ordinal_scope",
        "video_identity",
        "video_manifest_digest",
        "recording_frame_index_digest_algorithm",
        "recording_frame_index_digest",
        "empty_frame_media_resolution",
        "clips",
    }
    if set(value) != expected_fields:
        raise ValueError("clipped_binding has an unexpected field set")
    if (
        value.get("schema_id") != "palette.refined_detection.clipped_binding"
        or value.get("schema_version") != 1
        or value.get("camera_cardinality") != 1
        or value.get("clip_ordinal_scope")
        != "snapshot_global_within_single_camera"
        or value.get("recording_frame_index_digest_algorithm")
        != "sha256_canonical_rows_v1"
        or value.get("empty_frame_media_resolution")
        != "complete_frame_map_independent_of_rows"
    ):
        raise ValueError("clipped_binding schema, scope, or algorithm mismatch")
    raw_clips = value.get("clips")
    if not isinstance(raw_clips, list):
        raise ValueError("clipped_binding clips must be an ordered JSON array")
    clip_fields = {
        "clip_index",
        "clip_id",
        "media_identity",
        "media_digest_algorithm",
        "media_digest",
        "parent_frame_start",
        "parent_frame_stop",
        "frame_count",
        "frame_map_digest_algorithm",
        "frame_map_digest",
        "source_refined_run_id",
        "source_refined_manifest_digest",
    }
    clips: list[RefinedDetectionClipBinding] = []
    for index, raw_clip in enumerate(raw_clips):
        if not isinstance(raw_clip, Mapping) or set(raw_clip) != clip_fields:
            raise ValueError(f"clipped_binding clip {index} has an unexpected field set")
        if (
            raw_clip.get("media_digest_algorithm") != "sha256"
            or raw_clip.get("frame_map_digest_algorithm")
            != "sha256_canonical_rows_v1"
        ):
            raise ValueError(f"clipped_binding clip {index} digest algorithm mismatch")
        clip = RefinedDetectionClipBinding(
            clip_index=raw_clip.get("clip_index"),
            clip_id=raw_clip.get("clip_id"),
            media_identity=raw_clip.get("media_identity"),
            media_digest=raw_clip.get("media_digest"),
            parent_frame_start=raw_clip.get("parent_frame_start"),
            parent_frame_stop=raw_clip.get("parent_frame_stop"),
            frame_map_digest=raw_clip.get("frame_map_digest"),
            source_refined_run_id=raw_clip.get("source_refined_run_id"),
            source_refined_manifest_digest=raw_clip.get(
                "source_refined_manifest_digest"
            ),
        )
        if raw_clip.get("frame_count") != clip.frame_count:
            raise ValueError(f"clipped_binding clip {index} frame_count mismatch")
        clips.append(clip)
    parsed = RefinedDetectionClippedBinding(
        collection_id=value.get("collection_id"),
        collection_manifest_digest=value.get("collection_manifest_digest"),
        camera_serial=value.get("camera_serial"),
        video_identity=value.get("video_identity"),
        video_manifest_digest=value.get("video_manifest_digest"),
        recording_frame_index_digest=value.get("recording_frame_index_digest"),
        clips=tuple(clips),
    )
    if parsed.as_manifest() != dict(value):
        raise ValueError("clipped_binding is not in canonical persisted form")
    return parsed


def _dimensions_from_logical_schema(
    logical: Mapping[str, Any],
) -> RefinedDetectionDimensions:
    raw = logical.get("dimensions")
    expected_fields = {
        "n_frames",
        "n_instances",
        "n_source_detections",
        "n_frame_boundaries",
        "source_width",
        "source_height",
        "lineage_profile",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise ValueError("logical_schema dimensions have an unexpected field set")
    dimensions = RefinedDetectionDimensions(
        n_frames=raw.get("n_frames"),
        n_instances=raw.get("n_instances"),
        n_source_detections=raw.get("n_source_detections"),
        source_width=raw.get("source_width"),
        source_height=raw.get("source_height"),
        lineage_profile=raw.get("lineage_profile"),
    )
    if raw.get("n_frame_boundaries") != dimensions.n_frames + 1:
        raise ValueError("logical_schema n_frame_boundaries must equal n_frames + 1")
    return dimensions


def validate_refined_detection_run_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate the immutable document without opening external declarations."""

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
        resolved_dimensions: RefinedDetectionDimensions | None = None
        try:
            resolved_dimensions = _dimensions_from_logical_schema(logical)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        dimensions = logical.get("dimensions")
        profile = (
            resolved_dimensions.lineage_profile.value
            if resolved_dimensions is not None
            else (
                dimensions.get("lineage_profile")
                if isinstance(dimensions, Mapping)
                else None
            )
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
            else:
                try:
                    parsed_clipped = parse_refined_detection_clipped_binding(clipped)
                    if (
                        resolved_dimensions is not None
                        and parsed_clipped.n_frames
                        != resolved_dimensions.n_frames
                    ):
                        errors.append(
                            "clipped_binding intervals must cover logical n_frames"
                        )
                except (TypeError, ValueError) as exc:
                    errors.append(str(exc))
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
            errors.extend(
                _validate_reason_registry_document(
                    registry,
                    name=name,
                    expected_id=expected_id,
                )
            )
    return tuple(errors)


def validate_refined_detection_reason_code_coverage(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> tuple[str, ...]:
    """Require every persisted uint16 reason code to exist in its registry."""

    errors: list[str] = []
    payload = manifest.get("payload")
    registries = (
        payload.get("reason_registries")
        if isinstance(payload, Mapping)
        else None
    )
    if not isinstance(registries, Mapping):
        return ("reason registry coverage requires valid manifest registries",)
    for registry_name, path in (
        ("instances", "instances/reason_codes"),
        ("source_detections", "source_detections/reason_codes"),
    ):
        registry = registries.get(registry_name)
        codes = registry.get("codes") if isinstance(registry, Mapping) else None
        if not isinstance(codes, Mapping):
            errors.append(f"{registry_name} reason registry codes are unavailable")
            continue
        if path not in arrays:
            errors.append(f"reason registry coverage is missing array {path!r}")
            continue
        value = arrays[path]
        try:
            observed = np.asarray(value if isinstance(value, np.ndarray) else value[...])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"cannot read reason-code array {path!r}: {exc}")
            continue
        if observed.dtype != np.dtype("uint16") or observed.ndim != 1:
            errors.append(f"reason-code array {path!r} must be rank-1 uint16")
            continue
        unknown = sorted(
            int(code)
            for code in np.unique(observed).tolist()
            if str(int(code)) not in codes
        )
        if unknown:
            errors.append(
                f"reason-code array {path!r} contains unregistered codes {unknown!r}"
            )
    return tuple(errors)


def validate_refined_detection_publication(
    manifest: Mapping[str, Any],
    *,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Any],
) -> tuple[str, ...]:
    """Run the fail-closed gate required before a snapshot is contract-valid.

    This combines intrinsic manifest parsing with recomputation of the exact
    direct/consolidated metadata digest, complete logical-array validation, and
    persisted reason-code coverage.  A manifest-only parse is insufficient for
    publication or promotion.
    """

    errors = list(validate_refined_detection_run_manifest(manifest))
    payload = manifest.get("payload")
    logical = payload.get("logical_schema") if isinstance(payload, Mapping) else None
    if not isinstance(logical, Mapping):
        return (*errors, "publication validation requires logical_schema")
    try:
        dimensions = _dimensions_from_logical_schema(logical)
    except (TypeError, ValueError) as exc:
        return (*errors, f"publication dimensions are invalid: {exc}")
    try:
        observed_metadata_digest = refined_detection_metadata_declarations_digest(
            direct_metadata_declarations,
            consolidated_metadata_by_path=consolidated_metadata_declarations,
            dimensions=dimensions,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"metadata declaration validation failed: {exc}")
    else:
        publication = payload.get("publication")
        expected_metadata_digest = (
            publication.get("metadata_declarations_digest")
            if isinstance(publication, Mapping)
            else None
        )
        if observed_metadata_digest != expected_metadata_digest:
            errors.append("metadata_declarations_digest does not match declarations")

    clipped_binding: RefinedDetectionClippedBinding | None = None
    raw_clipped = logical.get("clipped_binding")
    if raw_clipped is not None:
        if not isinstance(raw_clipped, Mapping):
            errors.append("clipped_binding must be an object")
        else:
            try:
                clipped_binding = parse_refined_detection_clipped_binding(raw_clipped)
            except (TypeError, ValueError) as exc:
                errors.append(f"clipped_binding validation failed: {exc}")
    schema_issues = REFINED_DETECTION_SCHEMA_V1.validate(
        arrays,
        dimensions=dimensions,
        clipped_binding=clipped_binding,
    )
    errors.extend(
        f"array schema {issue.code} at {issue.path}: {issue.message}"
        for issue in schema_issues
    )
    errors.extend(validate_refined_detection_reason_code_coverage(manifest, arrays))
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
    if lineage.get("snapshot_id") == parent_lineage.get("snapshot_id"):
        errors.append("successor snapshot_id must differ from parent")
    parent_key_allocator = parent_lineage.get("manual_instance_key_allocator")
    if not isinstance(parent_key_allocator, Mapping):
        errors.append("parent manual_instance_key_allocator must be an object")
    elif not isinstance(manual_allocator, Mapping) or manual_allocator.get(
        "recording_identity"
    ) != parent_key_allocator.get("recording_identity"):
        errors.append("successor recording_identity differs from parent")
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
    "METADATA_DECLARATIONS_SCHEMA_ID",
    "METADATA_DECLARATIONS_SCHEMA_VERSION",
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
    "normalize_refined_detection_metadata_declarations",
    "parse_refined_detection_clipped_binding",
    "refined_detection_metadata_declarations_digest",
    "refined_detection_selection_contract_manifest",
    "validate_refined_detection_run_manifest",
    "validate_refined_detection_publication",
    "validate_refined_detection_reason_code_coverage",
    "validate_refined_detection_snapshot_identity",
    "validate_refined_detection_authority_provenance",
]
