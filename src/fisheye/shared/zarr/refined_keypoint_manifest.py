"""Strict manifest and publication gate for refined keypoint v2 snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence
from uuid import UUID

import numpy as np

from fisheye.shared.zarr.array_factory import (
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.keypoint_manifest import (
    keypoint_skeleton_digest,
    keypoint_skeleton_document,
    validate_keypoint_run_manifest,
)
from fisheye.shared.zarr.keypoint_quality_manifest import (
    validate_keypoint_quality_run_manifest,
)
from fisheye.shared.zarr.keypoint_schema import (
    KEYPOINT_SCHEMA_V2,
    REFINED_KEYPOINT_SCHEMA_V2,
    KeypointDimensions,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.refined_keypoint_storage import (
    RefinedKeypointStoragePlanSet,
    plan_refined_keypoint_storage,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr.training_keypoint_crop_source import (
    TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID,
)

REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_ID = "palette.refined_keypoint.run_manifest"
REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_VERSION = 1
REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
REFINED_KEYPOINT_RUN_MANIFEST_PERSISTED_PATH = (
    "refined_keypoints_runs/<run>/zarr.json.attributes.run_manifest"
)
REFINED_KEYPOINT_LOGICAL_CONTENT_SCHEMA_ID = "palette.refined_keypoint.logical_content"
REFINED_KEYPOINT_LOGICAL_CONTENT_SCHEMA_VERSION = 1
REFINED_KEYPOINT_SOURCE_BINDINGS_SCHEMA_ID = "palette.refined_keypoint.source_bindings"
REFINED_KEYPOINT_SOURCE_BINDINGS_SCHEMA_VERSION = 2
REFINED_KEYPOINT_SNAPSHOT_IDENTITY_SCHEMA_ID = (
    "palette.refined_keypoint.snapshot_identity"
)
REFINED_KEYPOINT_SNAPSHOT_IDENTITY_SCHEMA_VERSION = 1
REFINED_KEYPOINT_CODE_REGISTRIES_SCHEMA_ID = "palette.refined_keypoint.code_registries"
REFINED_KEYPOINT_CODE_REGISTRIES_SCHEMA_VERSION = 1
REFINED_KEYPOINT_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_bytes_v1"
REFINED_KEYPOINT_RETIRED_KEYS_DIGEST_ALGORITHM = (
    "sha256_sorted_unique_uint64_c_contiguous_bytes_v1"
)
REFINED_KEYPOINT_METADATA_DIGEST_SCOPE = (
    "exact_group_and_array_declarations_with_attributes_redacting_only_run_manifest"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
    return normalized


def _require_text(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    return normalized


def _require_run_id(value: object, *, name: str = "run_id") -> str:
    normalized = _require_text(value, name=name)
    if "/" in normalized:
        raise ValueError(f"{name} must be one archive group name.")
    return normalized


def _require_uuid(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    try:
        parsed = UUID(normalized)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a canonical UUID.") from exc
    if str(parsed) != normalized:
        raise ValueError(f"{name} must use canonical lowercase UUID form.")
    return normalized


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _manifest_payload(manifest: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} payload must be an object.")
    return payload


def _logical_content_document(
    payload: Mapping[str, Any], *, name: str
) -> Mapping[str, Any]:
    envelope = payload.get("logical_content")
    if not isinstance(envelope, Mapping):
        raise ValueError(f"{name} logical_content must be an object.")
    document = envelope.get("document")
    if not isinstance(document, Mapping):
        raise ValueError(f"{name} logical-content document must be an object.")
    return document


def _dimensions_from_manifest(value: object) -> KeypointDimensions:
    expected = {
        "n_frames",
        "n_frame_boundaries",
        "n_instances",
        "n_keypoints",
        "source_width",
        "source_height",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Refined-keypoint dimensions are not exact.")
    dimensions = KeypointDimensions(
        n_frames=value.get("n_frames"),
        n_instances=value.get("n_instances"),
        n_keypoints=value.get("n_keypoints"),
        source_width=value.get("source_width"),
        source_height=value.get("source_height"),
    )
    if dict(value) != dimensions.as_manifest():
        raise ValueError("Refined-keypoint dimensions are not canonical.")
    return dimensions


def _validated_skeleton_semantics(
    value: object,
    *,
    skeleton_id: str,
    skeleton_digest: str,
    dimensions: KeypointDimensions,
) -> dict[str, Any]:
    expected = {
        "schema_id",
        "schema_version",
        "skeleton_id",
        "kpt_shape",
        "keypoint_labels",
        "nodes",
        "edges",
        "heading_computation",
        "heading_computation_source",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Refined-keypoint skeleton semantics are not exact.")
    document = dict(value)
    if (
        document.get("schema_id") != "palette.keypoint.skeleton_semantics"
        or document.get("schema_version") != 1
        or document.get("skeleton_id") != skeleton_id
    ):
        raise ValueError("Refined-keypoint skeleton semantic identity mismatch.")
    labels = document.get("keypoint_labels")
    nodes = document.get("nodes")
    edges = document.get("edges")
    if (
        not isinstance(labels, list)
        or len(labels) != dimensions.n_keypoints
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Refined-keypoint ordered labels are invalid.")
    if document.get("kpt_shape") != [dimensions.n_keypoints, 2]:
        raise ValueError("Refined-keypoint skeleton shape is invalid.")
    if not isinstance(nodes, list) or nodes != [
        {"id": index, "name": label} for index, label in enumerate(labels)
    ]:
        raise ValueError("Refined-keypoint skeleton nodes differ from ordered labels.")
    if not isinstance(edges, list):
        raise ValueError("Refined-keypoint skeleton edges must be an array.")
    normalized_edges: list[tuple[int, int]] = []
    for edge in edges:
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(type(index) is not int for index in edge)
            or any(index < 0 or index >= dimensions.n_keypoints for index in edge)
        ):
            raise ValueError("Refined-keypoint skeleton edge is invalid.")
        normalized_edges.append((edge[0], edge[1]))
    if len(normalized_edges) != len(set(normalized_edges)):
        raise ValueError("Refined-keypoint skeleton edges cannot contain duplicates.")
    if not isinstance(document.get("heading_computation"), Mapping):
        raise ValueError("Refined-keypoint heading computation must be an object.")
    _require_text(
        document.get("heading_computation_source"),
        name="heading_computation_source",
    )
    canonical_json_bytes(document)
    if canonical_json_sha256(document) != skeleton_digest:
        raise ValueError("Refined-keypoint skeleton semantics digest mismatch.")
    return document


def normalized_retired_instance_keys(values: Sequence[int]) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.uint64)
    if result.ndim != 1:
        raise ValueError("retired_instance_keys must be one-dimensional.")
    if result.size and (
        np.any(result[1:] <= result[:-1])
        or len(set(int(value) for value in result)) != result.size
    ):
        raise ValueError("retired_instance_keys must be sorted and unique.")
    return np.ascontiguousarray(result)


def retired_instance_keys_digest(values: Sequence[int]) -> str:
    return sha256_array(normalized_retired_instance_keys(values))


@dataclass(frozen=True)
class RefinedKeypointSourceBindings:
    recording_identity: str
    raw_run_id: str
    raw_manifest_digest: str
    raw_logical_content_digest: str
    raw_row_signatures_digest: str
    quality_run_id: str
    quality_manifest_digest: str
    quality_logical_content_digest: str
    quality_profile_digest: str
    quality_source_row_signatures_digest: str
    crop_run_id: str
    crop_manifest_digest: str
    crop_logical_content_digest: str
    skeleton_id: str
    skeleton_digest: str
    skeleton_semantics: Mapping[str, Any]
    coordinate_catalog_digest: str
    dimensions: KeypointDimensions

    def __post_init__(self) -> None:
        for name in ("recording_identity", "skeleton_id"):
            object.__setattr__(
                self, name, _require_text(getattr(self, name), name=name)
            )
        for name in ("raw_run_id", "quality_run_id", "crop_run_id"):
            object.__setattr__(
                self, name, _require_run_id(getattr(self, name), name=name)
            )
        for name in (
            "raw_manifest_digest",
            "raw_logical_content_digest",
            "raw_row_signatures_digest",
            "quality_manifest_digest",
            "quality_logical_content_digest",
            "quality_profile_digest",
            "quality_source_row_signatures_digest",
            "crop_manifest_digest",
            "crop_logical_content_digest",
            "skeleton_digest",
            "coordinate_catalog_digest",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        if not isinstance(self.dimensions, KeypointDimensions):
            raise TypeError("dimensions must be KeypointDimensions.")
        object.__setattr__(
            self,
            "skeleton_semantics",
            _validated_skeleton_semantics(
                self.skeleton_semantics,
                skeleton_id=self.skeleton_id,
                skeleton_digest=self.skeleton_digest,
                dimensions=self.dimensions,
            ),
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": REFINED_KEYPOINT_SOURCE_BINDINGS_SCHEMA_ID,
            "schema_version": REFINED_KEYPOINT_SOURCE_BINDINGS_SCHEMA_VERSION,
            "recording_identity": self.recording_identity,
            "raw_keypoint_snapshot": {
                "stage": "keypoints",
                "run_id": self.raw_run_id,
                "run_path": f"keypoints_runs/{self.raw_run_id}",
                "schema_id": KEYPOINT_SCHEMA_V2.schema_id,
                "schema_version": KEYPOINT_SCHEMA_V2.schema_version,
                "manifest_digest": self.raw_manifest_digest,
                "logical_content_digest": self.raw_logical_content_digest,
                "keypoint_row_signatures_digest": self.raw_row_signatures_digest,
            },
            "quality_snapshot": {
                "stage": "keypoint_quality",
                "run_id": self.quality_run_id,
                "run_path": f"keypoint_quality_runs/{self.quality_run_id}",
                "schema_id": "palette.stage.keypoint_quality",
                "schema_version": 1,
                "manifest_digest": self.quality_manifest_digest,
                "logical_content_digest": self.quality_logical_content_digest,
                "profile_digest": self.quality_profile_digest,
                "source_row_signatures_digest": (
                    self.quality_source_row_signatures_digest
                ),
            },
            "crop_snapshot": {
                "stage": "crop",
                "run_id": self.crop_run_id,
                "run_path": f"crop_runs/{self.crop_run_id}",
                "manifest_digest": self.crop_manifest_digest,
                "logical_content_digest": self.crop_logical_content_digest,
            },
            "skeleton": {
                "skeleton_id": self.skeleton_id,
                "skeleton_digest": self.skeleton_digest,
                "semantics": dict(self.skeleton_semantics),
            },
            "coordinate_catalog_digest": self.coordinate_catalog_digest,
            "dimensions": self.dimensions.as_manifest(),
        }


@dataclass(frozen=True)
class RefinedKeypointSnapshotIdentity:
    recording_identity: str
    lineage_id: str
    snapshot_id: str
    parent_run_id: str | None = None
    parent_manifest_digest: str | None = None
    parent_snapshot_id: str | None = None
    ancestry_snapshot_ids: tuple[str, ...] = ()
    retired_instance_key_count: int = 0
    retired_instance_keys_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "recording_identity",
            _require_text(self.recording_identity, name="recording_identity"),
        )
        object.__setattr__(
            self, "lineage_id", _require_uuid(self.lineage_id, name="lineage_id")
        )
        object.__setattr__(
            self, "snapshot_id", _require_uuid(self.snapshot_id, name="snapshot_id")
        )
        parents = (
            self.parent_run_id,
            self.parent_manifest_digest,
            self.parent_snapshot_id,
        )
        if any(value is None for value in parents) and not all(
            value is None for value in parents
        ):
            raise ValueError(
                "Parent identity fields must be all present or all absent."
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
                    self.parent_manifest_digest, name="parent_manifest_digest"
                ),
            )
            object.__setattr__(
                self,
                "parent_snapshot_id",
                _require_uuid(self.parent_snapshot_id, name="parent_snapshot_id"),
            )
        ancestry = tuple(
            _require_uuid(value, name="ancestry_snapshot_id")
            for value in self.ancestry_snapshot_ids
        )
        if len(ancestry) != len(set(ancestry)):
            raise ValueError("ancestry_snapshot_ids cannot contain duplicates.")
        if self.snapshot_id in ancestry:
            raise ValueError("A snapshot cannot appear in its own ancestry.")
        if self.parent_snapshot_id is None:
            if ancestry:
                raise ValueError("An initial snapshot cannot declare ancestry.")
        elif not ancestry or ancestry[-1] != self.parent_snapshot_id:
            raise ValueError("Successor ancestry must end with parent_snapshot_id.")
        object.__setattr__(self, "ancestry_snapshot_ids", ancestry)
        if (
            type(self.retired_instance_key_count) is not int
            or self.retired_instance_key_count < 0
        ):
            raise ValueError("retired_instance_key_count must be nonnegative.")
        object.__setattr__(
            self,
            "retired_instance_keys_digest",
            _require_sha256(
                self.retired_instance_keys_digest,
                name="retired_instance_keys_digest",
            ),
        )

    def as_manifest(self) -> dict[str, object]:
        parent = None
        if self.parent_run_id is not None:
            parent = {
                "run_id": self.parent_run_id,
                "manifest_digest": self.parent_manifest_digest,
                "snapshot_id": self.parent_snapshot_id,
            }
        return {
            "schema_id": REFINED_KEYPOINT_SNAPSHOT_IDENTITY_SCHEMA_ID,
            "schema_version": REFINED_KEYPOINT_SNAPSHOT_IDENTITY_SCHEMA_VERSION,
            "recording_identity": self.recording_identity,
            "lineage_id": self.lineage_id,
            "snapshot_id": self.snapshot_id,
            "parent": parent,
            "ancestry_snapshot_ids": list(self.ancestry_snapshot_ids),
            "instance_key_policy": "preserve_raw_observation_identity_v1",
            "retired_instance_keys": {
                "count": self.retired_instance_key_count,
                "digest_algorithm": REFINED_KEYPOINT_RETIRED_KEYS_DIGEST_ALGORITHM,
                "digest": self.retired_instance_keys_digest,
                "nonreuse": True,
            },
        }


def initial_refined_keypoint_snapshot_identity(
    *,
    recording_identity: str,
    lineage_id: str,
    snapshot_id: str,
    retired_instance_keys: Sequence[int] = (),
) -> RefinedKeypointSnapshotIdentity:
    retired = normalized_retired_instance_keys(retired_instance_keys)
    return RefinedKeypointSnapshotIdentity(
        recording_identity=recording_identity,
        lineage_id=lineage_id,
        snapshot_id=snapshot_id,
        retired_instance_key_count=int(retired.size),
        retired_instance_keys_digest=sha256_array(retired),
    )


def _source_bindings_from_manifest(
    value: Mapping[str, Any],
) -> RefinedKeypointSourceBindings:
    expected = {
        "schema_id",
        "schema_version",
        "recording_identity",
        "raw_keypoint_snapshot",
        "quality_snapshot",
        "crop_snapshot",
        "skeleton",
        "coordinate_catalog_digest",
        "dimensions",
    }
    if set(value) != expected:
        raise ValueError("Refined-keypoint source bindings have unexpected fields.")
    raw = value.get("raw_keypoint_snapshot")
    quality = value.get("quality_snapshot")
    crop = value.get("crop_snapshot")
    skeleton = value.get("skeleton")
    if not all(isinstance(item, Mapping) for item in (raw, quality, crop, skeleton)):
        raise TypeError("Refined-keypoint nested source bindings must be objects.")
    source = RefinedKeypointSourceBindings(
        recording_identity=value.get("recording_identity"),
        raw_run_id=raw.get("run_id"),
        raw_manifest_digest=raw.get("manifest_digest"),
        raw_logical_content_digest=raw.get("logical_content_digest"),
        raw_row_signatures_digest=raw.get("keypoint_row_signatures_digest"),
        quality_run_id=quality.get("run_id"),
        quality_manifest_digest=quality.get("manifest_digest"),
        quality_logical_content_digest=quality.get("logical_content_digest"),
        quality_profile_digest=quality.get("profile_digest"),
        quality_source_row_signatures_digest=quality.get(
            "source_row_signatures_digest"
        ),
        crop_run_id=crop.get("run_id"),
        crop_manifest_digest=crop.get("manifest_digest"),
        crop_logical_content_digest=crop.get("logical_content_digest"),
        skeleton_id=skeleton.get("skeleton_id"),
        skeleton_digest=skeleton.get("skeleton_digest"),
        skeleton_semantics=skeleton.get("semantics"),
        coordinate_catalog_digest=value.get("coordinate_catalog_digest"),
        dimensions=_dimensions_from_manifest(value.get("dimensions")),
    )
    if dict(value) != source.as_manifest():
        raise ValueError("Refined-keypoint source bindings are not canonical.")
    return source


def refined_keypoint_source_bindings_from_manifest(
    value: Mapping[str, Any],
) -> RefinedKeypointSourceBindings:
    """Parse exact, digest-bound refined-keypoint source semantics."""

    return _source_bindings_from_manifest(value)


def refined_keypoint_snapshot_identity_from_manifest(
    manifest: Mapping[str, Any],
) -> RefinedKeypointSnapshotIdentity:
    """Parse the exact snapshot identity from one persisted run manifest."""

    payload = _manifest_payload(manifest, name="refined keypoint manifest")
    value = payload.get("snapshot_identity")
    if not isinstance(value, Mapping):
        raise TypeError("Refined-keypoint snapshot identity must be an object.")
    return _identity_from_manifest(value)


def _identity_from_manifest(
    value: Mapping[str, Any],
) -> RefinedKeypointSnapshotIdentity:
    expected = {
        "schema_id",
        "schema_version",
        "recording_identity",
        "lineage_id",
        "snapshot_id",
        "parent",
        "ancestry_snapshot_ids",
        "instance_key_policy",
        "retired_instance_keys",
    }
    if set(value) != expected:
        raise ValueError("Refined-keypoint snapshot identity has unexpected fields.")
    parent = value.get("parent")
    if parent is not None and (
        not isinstance(parent, Mapping)
        or set(parent) != {"run_id", "manifest_digest", "snapshot_id"}
    ):
        raise ValueError("Refined-keypoint parent identity is invalid.")
    retired = value.get("retired_instance_keys")
    if not isinstance(retired, Mapping) or set(retired) != {
        "count",
        "digest_algorithm",
        "digest",
        "nonreuse",
    }:
        raise ValueError("Refined-keypoint retired-key identity is invalid.")
    if (
        value.get("schema_id") != REFINED_KEYPOINT_SNAPSHOT_IDENTITY_SCHEMA_ID
        or value.get("schema_version")
        != REFINED_KEYPOINT_SNAPSHOT_IDENTITY_SCHEMA_VERSION
        or value.get("instance_key_policy") != "preserve_raw_observation_identity_v1"
        or retired.get("digest_algorithm")
        != REFINED_KEYPOINT_RETIRED_KEYS_DIGEST_ALGORITHM
        or retired.get("nonreuse") is not True
    ):
        raise ValueError("Refined-keypoint snapshot identity policy mismatch.")
    ancestry = value.get("ancestry_snapshot_ids")
    if not isinstance(ancestry, list):
        raise TypeError("ancestry_snapshot_ids must be an array.")
    identity = RefinedKeypointSnapshotIdentity(
        recording_identity=value.get("recording_identity"),
        lineage_id=value.get("lineage_id"),
        snapshot_id=value.get("snapshot_id"),
        parent_run_id=None if parent is None else parent.get("run_id"),
        parent_manifest_digest=(
            None if parent is None else parent.get("manifest_digest")
        ),
        parent_snapshot_id=None if parent is None else parent.get("snapshot_id"),
        ancestry_snapshot_ids=tuple(ancestry),
        retired_instance_key_count=retired.get("count"),
        retired_instance_keys_digest=retired.get("digest"),
    )
    if dict(value) != identity.as_manifest():
        raise ValueError("Refined-keypoint snapshot identity is not canonical.")
    return identity


def _code_map(value: Mapping[int, str], *, name: str, maximum: int) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_code, raw_label in value.items():
        if type(raw_code) is not int or not (0 <= raw_code <= maximum):
            raise ValueError(f"{name} codes must be exact integers in range.")
        label = str(raw_label).strip()
        if not _CODE_LABEL.fullmatch(label):
            raise ValueError(f"{name} labels must be canonical lowercase identifiers.")
        normalized[raw_code] = label
    if 0 not in normalized:
        raise ValueError(f"{name} must define code zero.")
    if len(set(normalized.values())) != len(normalized):
        raise ValueError(f"{name} labels must be unique.")
    return dict(sorted(normalized.items()))


def _code_registry_document(
    review_state_map: Mapping[int, str], reason_code_map: Mapping[int, str]
) -> dict[str, object]:
    review = _code_map(review_state_map, name="review_state_map", maximum=255)
    reason = _code_map(reason_code_map, name="reason_code_map", maximum=65535)
    document = {
        "schema_id": REFINED_KEYPOINT_CODE_REGISTRIES_SCHEMA_ID,
        "schema_version": REFINED_KEYPOINT_CODE_REGISTRIES_SCHEMA_VERSION,
        "review_state_map": {str(code): label for code, label in review.items()},
        "reason_code_map": {str(code): label for code, label in reason.items()},
        "zero_code_semantics": {
            "review_state": review[0],
            "reason": reason[0],
        },
    }
    return {
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "digest": canonical_json_sha256(document),
        "document": document,
    }


def _code_registries_from_manifest(
    value: Mapping[str, Any],
) -> tuple[dict[int, str], dict[int, str]]:
    if set(value) != {"digest_algorithm", "digest", "document"}:
        raise ValueError("Refined-keypoint code-registry envelope is invalid.")
    document = value.get("document")
    if not isinstance(document, Mapping) or set(document) != {
        "schema_id",
        "schema_version",
        "review_state_map",
        "reason_code_map",
        "zero_code_semantics",
    }:
        raise ValueError("Refined-keypoint code-registry document is invalid.")
    if (
        value.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or value.get("digest") != canonical_json_sha256(document)
        or document.get("schema_id") != REFINED_KEYPOINT_CODE_REGISTRIES_SCHEMA_ID
        or document.get("schema_version")
        != REFINED_KEYPOINT_CODE_REGISTRIES_SCHEMA_VERSION
    ):
        raise ValueError("Refined-keypoint code-registry identity/digest mismatch.")

    def parse(raw: object, *, name: str, maximum: int) -> dict[int, str]:
        if not isinstance(raw, Mapping):
            raise TypeError(f"{name} must be an object.")
        parsed: dict[int, str] = {}
        for key, label in raw.items():
            text = str(key)
            try:
                code = int(text)
            except ValueError as exc:
                raise ValueError(f"{name} keys must be decimal integers.") from exc
            if str(code) != text:
                raise ValueError(f"{name} keys must be canonical decimal integers.")
            parsed[code] = str(label)
        return _code_map(parsed, name=name, maximum=maximum)

    review = parse(
        document.get("review_state_map"), name="review_state_map", maximum=255
    )
    reason = parse(
        document.get("reason_code_map"), name="reason_code_map", maximum=65535
    )
    expected = _code_registry_document(review, reason)
    if dict(value) != expected:
        raise ValueError("Refined-keypoint code registries differ from frozen builder.")
    return review, reason


def refined_keypoint_code_maps_from_manifest(
    manifest: Mapping[str, Any],
) -> tuple[dict[int, str], dict[int, str]]:
    """Parse both exact code maps from one persisted run manifest."""

    payload = _manifest_payload(manifest, name="refined keypoint manifest")
    value = payload.get("code_registries")
    if not isinstance(value, Mapping):
        raise TypeError("Refined-keypoint code registries must be an object.")
    return _code_registries_from_manifest(value)


def build_refined_keypoint_source_bindings(
    *,
    raw_manifest: Mapping[str, Any],
    quality_manifest: Mapping[str, Any],
    crop_manifest: Mapping[str, Any],
) -> RefinedKeypointSourceBindings:
    raw_errors = validate_keypoint_run_manifest(raw_manifest)
    quality_errors = validate_keypoint_quality_run_manifest(quality_manifest)
    if raw_errors:
        raise ValueError("Raw keypoint manifest is invalid: " + "; ".join(raw_errors))
    if quality_errors:
        raise ValueError("Quality manifest is invalid: " + "; ".join(quality_errors))
    raw_payload = _manifest_payload(raw_manifest, name="raw keypoint manifest")
    quality_payload = _manifest_payload(quality_manifest, name="quality manifest")
    crop_payload = _manifest_payload(crop_manifest, name="crop manifest")
    raw_logical = raw_payload.get("logical_schema")
    raw_source_crop = raw_payload.get("source_crop_snapshot")
    raw_content = _logical_content_document(raw_payload, name="raw keypoint")
    quality_logical = quality_payload.get("logical_schema")
    quality_source = quality_payload.get("source_keypoint_snapshot")
    _logical_content_document(quality_payload, name="quality")
    _logical_content_document(crop_payload, name="crop")
    coordinate = raw_payload.get("coordinate_contract")
    pose_binding = raw_payload.get("pose_model_schema_binding")
    if not all(
        isinstance(item, Mapping)
        for item in (
            raw_logical,
            raw_source_crop,
            quality_logical,
            quality_source,
            coordinate,
            pose_binding,
        )
    ):
        raise ValueError("Source manifests lack required nested declarations.")
    dimensions = _dimensions_from_manifest(raw_logical.get("dimensions"))
    raw_digest = canonical_json_sha256(raw_manifest)
    quality_digest = canonical_json_sha256(quality_manifest)
    crop_digest = canonical_json_sha256(crop_manifest)
    if quality_source.get("manifest_digest") != raw_digest:
        raise ValueError("Quality source does not bind the supplied raw manifest.")
    if raw_source_crop.get("manifest_digest") != crop_digest:
        raise ValueError("Raw keypoints do not bind the supplied crop manifest.")
    crop_logical_envelope = crop_payload.get("logical_content")
    if not isinstance(crop_logical_envelope, Mapping):
        raise ValueError("Crop logical-content envelope is missing.")
    crop_logical_digest = _require_sha256(
        crop_logical_envelope.get("digest"), name="crop_logical_content_digest"
    )
    if raw_source_crop.get("logical_content_digest") != crop_logical_digest:
        raise ValueError("Raw keypoints do not bind the crop logical content.")
    if quality_source.get("keypoint_row_signatures_digest") != raw_content.get(
        "arrays", {}
    ).get("keypoint_row_signature", {}).get("sha256"):
        raise ValueError("Quality and raw row-signature identities differ.")
    if crop_manifest.get("schema_id") == TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID:
        recording = _require_text(
            crop_payload.get("recording_identity"), name="recording_identity"
        )
    else:
        crop_refined = crop_payload.get("source_refined_snapshot")
        crop_pixel = crop_payload.get("source_pixel_authority")
        if not isinstance(crop_refined, Mapping) or not isinstance(
            crop_pixel, Mapping
        ):
            raise ValueError("Crop recording identity evidence is incomplete.")
        recording = _require_text(
            crop_refined.get("recording_identity"), name="recording_identity"
        )
        if crop_pixel.get("recording_identity") != recording:
            raise ValueError("Crop refined and pixel recording identities differ.")
    pose_schema = pose_binding.get("pose_schema")
    if not isinstance(pose_schema, Mapping):
        raise ValueError("Raw pose schema is missing.")
    skeleton_id = _require_text(pose_schema.get("skeleton_id"), name="skeleton_id")
    skeleton_digest = keypoint_skeleton_digest(pose_binding)
    skeleton_semantics = keypoint_skeleton_document(pose_binding)
    quality_profile = quality_logical.get("profile")
    if not isinstance(quality_profile, Mapping):
        raise ValueError("Quality profile is missing.")
    return RefinedKeypointSourceBindings(
        recording_identity=recording,
        raw_run_id=raw_payload.get("run_id"),
        raw_manifest_digest=raw_digest,
        raw_logical_content_digest=raw_payload["logical_content"]["digest"],
        raw_row_signatures_digest=raw_content["arrays"]["keypoint_row_signature"][
            "sha256"
        ],
        quality_run_id=quality_payload.get("run_id"),
        quality_manifest_digest=quality_digest,
        quality_logical_content_digest=quality_payload["logical_content"]["digest"],
        quality_profile_digest=quality_profile.get("profile_digest"),
        quality_source_row_signatures_digest=quality_source.get(
            "keypoint_row_signatures_digest"
        ),
        crop_run_id=raw_source_crop.get("run_id"),
        crop_manifest_digest=crop_digest,
        crop_logical_content_digest=crop_logical_digest,
        skeleton_id=skeleton_id,
        skeleton_digest=skeleton_digest,
        skeleton_semantics=skeleton_semantics,
        coordinate_catalog_digest=coordinate.get("digest"),
        dimensions=dimensions,
    )


def refined_keypoint_logical_content_document(
    arrays: Mapping[str, Any],
    *,
    dimensions: KeypointDimensions,
    source: RefinedKeypointSourceBindings,
    identity: RefinedKeypointSnapshotIdentity,
    source_crop_arrays: Mapping[str, Any],
    review_state_map: Mapping[int, str],
    reason_code_map: Mapping[int, str],
) -> dict[str, object]:
    REFINED_KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=source.skeleton_digest,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    declarations: dict[str, object] = {}
    for path in REFINED_KEYPOINT_SCHEMA_V2.binding_paths:
        value = _array_values(arrays[path])
        declarations[path] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "digest_algorithm": REFINED_KEYPOINT_ARRAY_DIGEST_ALGORITHM,
            "sha256": sha256_array(value),
        }
    return {
        "schema_id": REFINED_KEYPOINT_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": REFINED_KEYPOINT_LOGICAL_CONTENT_SCHEMA_VERSION,
        "logical_schema": {
            "id": REFINED_KEYPOINT_SCHEMA_V2.schema_id,
            "version": REFINED_KEYPOINT_SCHEMA_V2.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "source_manifest_digests": {
            "raw_keypoints": source.raw_manifest_digest,
            "keypoint_quality": source.quality_manifest_digest,
            "crop": source.crop_manifest_digest,
        },
        "source_row_signatures_digest": source.raw_row_signatures_digest,
        "lineage_id": identity.lineage_id,
        "snapshot_id": identity.snapshot_id,
        "arrays": declarations,
    }


def refined_keypoint_metadata_declarations_digest(
    direct_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_by_path: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = {"", *REFINED_KEYPOINT_SCHEMA_V2.binding_paths}
    if set(direct_by_path) != expected or set(consolidated_by_path) != expected:
        raise ValueError("Refined-keypoint metadata declaration paths are incomplete.")
    normalized: dict[str, object] = {}
    for path in sorted(expected):
        direct = metadata_without_empty_group_consolidation(
            direct_by_path[path], path=path
        )
        consolidated = metadata_without_empty_group_consolidation(
            consolidated_by_path[path], path=path
        )
        if path == "":
            for declaration in (direct, consolidated):
                attributes = declaration.get("attributes")
                if not isinstance(attributes, Mapping):
                    raise ValueError("Refined run group attributes must be an object.")
                redacted = dict(attributes)
                redacted.pop(REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE, None)
                declaration["attributes"] = redacted
        if direct != consolidated:
            raise ValueError(
                f"Direct and consolidated metadata differ at {path or '<run>'!r}."
            )
        normalized[path] = direct
    return canonical_json_sha256(
        {
            "scope": REFINED_KEYPOINT_METADATA_DIGEST_SCOPE,
            "declarations": normalized,
        }
    )


def build_refined_keypoint_run_manifest(
    *,
    run_id: str,
    dimensions: KeypointDimensions,
    source: RefinedKeypointSourceBindings,
    raw_manifest: Mapping[str, Any],
    quality_manifest: Mapping[str, Any],
    crop_manifest: Mapping[str, Any],
    storage_plan: RefinedKeypointStoragePlanSet,
    identity: RefinedKeypointSnapshotIdentity,
    arrays: Mapping[str, Any],
    source_crop_arrays: Mapping[str, Any],
    review_state_map: Mapping[int, str],
    reason_code_map: Mapping[int, str],
    retired_instance_keys: Sequence[int],
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    resolved_run_id = _require_run_id(run_id)
    observed_source = build_refined_keypoint_source_bindings(
        raw_manifest=raw_manifest,
        quality_manifest=quality_manifest,
        crop_manifest=crop_manifest,
    )
    if observed_source != source:
        raise ValueError("Refined source bindings differ from supplied manifests.")
    if source.dimensions != dimensions or storage_plan.dimensions != dimensions:
        raise ValueError("Refined source/storage dimensions differ.")
    if identity.recording_identity != source.recording_identity:
        raise ValueError("Refined snapshot recording identity differs from its source.")
    retired = normalized_retired_instance_keys(retired_instance_keys)
    if identity.retired_instance_key_count != int(
        retired.size
    ) or identity.retired_instance_keys_digest != sha256_array(retired):
        raise ValueError("Refined retired-key evidence differs from snapshot identity.")
    keys = _array_values(arrays["instance_key"])
    if np.intersect1d(keys, retired).size:
        raise ValueError("A live instance_key is also declared retired.")
    registries = _code_registry_document(review_state_map, reason_code_map)
    content = refined_keypoint_logical_content_document(
        arrays,
        dimensions=dimensions,
        source=source,
        identity=identity,
        source_crop_arrays=source_crop_arrays,
        review_state_map=review_state_map,
        reason_code_map=reason_code_map,
    )
    metadata_digest = refined_keypoint_metadata_declarations_digest(
        direct_metadata_declarations,
        consolidated_by_path=consolidated_metadata_declarations,
    )
    payload: dict[str, object] = {
        "run_id": resolved_run_id,
        "stage": "refined_keypoints",
        "publication": {
            "artifact_class": "reviewed_keypoint_authority_candidate",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": REFINED_KEYPOINT_METADATA_DIGEST_SCOPE,
            "metadata_declarations_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "metadata_declarations_digest": metadata_digest,
        },
        "logical_schema": REFINED_KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions),
        "storage_plan": storage_plan.as_manifest(),
        "source_bindings": source.as_manifest(),
        "snapshot_identity": identity.as_manifest(),
        "code_registries": registries,
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(content),
            "document": content,
        },
    }
    envelope = {
        "schema_id": REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
        "persisted_path": REFINED_KEYPOINT_RUN_MANIFEST_PERSISTED_PATH,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def _parse_manifest_components(manifest: Mapping[str, Any]):  # type: ignore[no-untyped-def]
    errors: list[str] = []
    expected_envelope = {
        "schema_id",
        "schema_version",
        "persisted_attribute",
        "persisted_path",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }
    if set(manifest) != expected_envelope:
        errors.append("refined-keypoint manifest envelope has unexpected fields")
    if (
        manifest.get("schema_id") != REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version")
        != REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("persisted_attribute")
        != REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE
        or manifest.get("persisted_path")
        != REFINED_KEYPOINT_RUN_MANIFEST_PERSISTED_PATH
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("refined-keypoint manifest envelope identity mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (
            [*errors, "refined-keypoint payload must be an object"],
            None,
            None,
            None,
            None,
            None,
            None,
        )
    try:
        digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"refined-keypoint manifest is not strict JSON: {exc}")
    else:
        if manifest.get("payload_digest") != digest:
            errors.append("refined-keypoint payload_digest mismatch")
    expected_payload = {
        "run_id",
        "stage",
        "publication",
        "logical_schema",
        "storage_plan",
        "source_bindings",
        "snapshot_identity",
        "code_registries",
        "logical_content",
    }
    if set(payload) != expected_payload:
        errors.append("refined-keypoint payload has unexpected fields")
    try:
        _require_run_id(payload.get("run_id"))
    except ValueError as exc:
        errors.append(str(exc))
    if payload.get("stage") != "refined_keypoints":
        errors.append("refined-keypoint stage mismatch")
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("refined-keypoint publication must be an object")
    else:
        expected_publication = {
            "artifact_class": "reviewed_keypoint_authority_candidate",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": REFINED_KEYPOINT_METADATA_DIGEST_SCOPE,
            "metadata_declarations_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "metadata_declarations_digest": publication.get(
                "metadata_declarations_digest"
            ),
        }
        if dict(publication) != expected_publication:
            errors.append("refined-keypoint publication is not exact")
        try:
            _require_sha256(
                publication.get("metadata_declarations_digest"),
                name="metadata_declarations_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))
    dimensions = None
    logical = payload.get("logical_schema")
    if not isinstance(logical, Mapping):
        errors.append("refined-keypoint logical_schema must be an object")
    else:
        try:
            dimensions = _dimensions_from_manifest(logical.get("dimensions"))
            if dict(logical) != REFINED_KEYPOINT_SCHEMA_V2.as_manifest(
                dimensions=dimensions
            ):
                errors.append("refined-keypoint logical_schema differs from builder")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    source = None
    source_value = payload.get("source_bindings")
    if isinstance(source_value, Mapping):
        try:
            source = _source_bindings_from_manifest(source_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    else:
        errors.append("refined-keypoint source_bindings must be an object")
    identity = None
    identity_value = payload.get("snapshot_identity")
    if isinstance(identity_value, Mapping):
        try:
            identity = _identity_from_manifest(identity_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    else:
        errors.append("refined-keypoint snapshot_identity must be an object")
    review = reason = None
    registry_value = payload.get("code_registries")
    if isinstance(registry_value, Mapping):
        try:
            review, reason = _code_registries_from_manifest(registry_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    else:
        errors.append("refined-keypoint code_registries must be an object")
    if (
        source is not None
        and dimensions is not None
        and source.dimensions != dimensions
    ):
        errors.append("refined-keypoint source dimensions mismatch")
    if (
        source is not None
        and identity is not None
        and source.recording_identity != identity.recording_identity
    ):
        errors.append("refined-keypoint recording identities mismatch")
    return errors, payload, dimensions, source, identity, review, reason


def validate_refined_keypoint_run_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    errors, payload, dimensions, source, identity, review, reason = (
        _parse_manifest_components(manifest)
    )
    if payload is None:
        return tuple(errors)
    storage = payload.get("storage_plan")
    if not isinstance(storage, Mapping):
        errors.append("refined-keypoint storage_plan must be an object")
    elif dimensions is not None:
        raw_profile = storage.get("storage_profile")
        if not isinstance(raw_profile, Mapping):
            errors.append("refined-keypoint storage profile must be an object")
        else:
            try:
                profile = storage_profile_from_manifest(raw_profile)
                expected = plan_refined_keypoint_storage(
                    dimensions, profile=profile
                ).as_manifest()
            except (TypeError, ValueError) as exc:
                errors.append(f"cannot reconstruct refined storage plan: {exc}")
            else:
                if dict(storage) != expected:
                    errors.append("refined-keypoint storage plan differs from planner")
    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping) or set(logical_content) != {
        "digest_algorithm",
        "digest",
        "document",
    }:
        errors.append("refined-keypoint logical_content envelope is invalid")
        return tuple(errors)
    document = logical_content.get("document")
    if not isinstance(document, Mapping):
        errors.append("refined-keypoint logical_content document must be an object")
        return tuple(errors)
    if logical_content.get(
        "digest_algorithm"
    ) != CANONICAL_JSON_DIGEST_ALGORITHM or logical_content.get(
        "digest"
    ) != canonical_json_sha256(
        document
    ):
        errors.append("refined-keypoint logical_content digest mismatch")
    expected_document_fields = {
        "schema_id",
        "schema_version",
        "logical_schema",
        "dimensions",
        "source_manifest_digests",
        "source_row_signatures_digest",
        "lineage_id",
        "snapshot_id",
        "arrays",
    }
    if set(document) != expected_document_fields:
        errors.append("refined-keypoint logical_content has unexpected fields")
    if (
        document.get("schema_id") != REFINED_KEYPOINT_LOGICAL_CONTENT_SCHEMA_ID
        or document.get("schema_version")
        != REFINED_KEYPOINT_LOGICAL_CONTENT_SCHEMA_VERSION
        or document.get("logical_schema")
        != {
            "id": REFINED_KEYPOINT_SCHEMA_V2.schema_id,
            "version": REFINED_KEYPOINT_SCHEMA_V2.schema_version,
        }
    ):
        errors.append("refined-keypoint logical_content identity mismatch")
    if (
        dimensions is not None
        and document.get("dimensions") != dimensions.as_manifest()
    ):
        errors.append("refined-keypoint logical_content dimensions mismatch")
    if source is not None:
        if document.get("source_manifest_digests") != {
            "raw_keypoints": source.raw_manifest_digest,
            "keypoint_quality": source.quality_manifest_digest,
            "crop": source.crop_manifest_digest,
        }:
            errors.append("refined-keypoint logical source digests mismatch")
        if (
            document.get("source_row_signatures_digest")
            != source.raw_row_signatures_digest
        ):
            errors.append("refined-keypoint logical source row digest mismatch")
    if identity is not None and (
        document.get("lineage_id") != identity.lineage_id
        or document.get("snapshot_id") != identity.snapshot_id
    ):
        errors.append("refined-keypoint logical snapshot identity mismatch")
    array_docs = document.get("arrays")
    if not isinstance(array_docs, Mapping) or set(array_docs) != set(
        REFINED_KEYPOINT_SCHEMA_V2.binding_paths
    ):
        errors.append("refined-keypoint logical array declarations mismatch")
    else:
        bindings = {
            binding.path: binding for binding in REFINED_KEYPOINT_SCHEMA_V2.bindings
        }
        for path, item in array_docs.items():
            if not isinstance(item, Mapping) or set(item) != {
                "shape",
                "dtype",
                "digest_algorithm",
                "sha256",
            }:
                errors.append(f"refined-keypoint array declaration invalid at {path}")
                continue
            if dimensions is not None:
                binding = bindings[path]
                contract = REFINED_KEYPOINT_SCHEMA_V2.contracts.resolve(
                    binding.contract_id, binding.contract_version
                )
                expected_shape = [
                    (
                        axis
                        if isinstance(axis, int)
                        else dimensions.contract_dimensions[axis]
                    )
                    for axis in contract.shape_template
                ]
                if item.get("shape") != expected_shape:
                    errors.append(f"refined-keypoint shape mismatch at {path}")
                if item.get("dtype") != str(contract.dtype.numpy_dtype):
                    errors.append(f"refined-keypoint dtype mismatch at {path}")
            if item.get("digest_algorithm") != REFINED_KEYPOINT_ARRAY_DIGEST_ALGORITHM:
                errors.append(f"refined-keypoint digest algorithm mismatch at {path}")
            try:
                _require_sha256(item.get("sha256"), name=f"{path} sha256")
            except ValueError as exc:
                errors.append(str(exc))
    if review is None or reason is None:
        errors.append("refined-keypoint code registries could not be reconstructed")
    return tuple(errors)


def validate_refined_keypoint_snapshot_identity(
    identity: RefinedKeypointSnapshotIdentity,
    *,
    arrays: Mapping[str, Any],
    raw_arrays: Mapping[str, Any],
    retired_instance_keys: Sequence[int],
) -> tuple[str, ...]:
    errors: list[str] = []
    retired = normalized_retired_instance_keys(retired_instance_keys)
    if identity.retired_instance_key_count != int(retired.size):
        errors.append("retired instance-key count mismatch")
    if identity.retired_instance_keys_digest != sha256_array(retired):
        errors.append("retired instance-key digest mismatch")
    try:
        keys = _array_values(arrays["instance_key"])
        raw_keys = _array_values(raw_arrays["instance_key"])
    except KeyError as exc:
        return (*errors, f"missing instance-key identity array: {exc}")
    if not np.array_equal(keys, raw_keys):
        errors.append("initial refined instance_key must exactly preserve raw order")
    if np.intersect1d(keys, retired).size:
        errors.append("live and retired instance keys overlap")
    if identity.parent_run_id is not None:
        errors.append(
            "successor identity validation is deferred to the delta compactor"
        )
    return tuple(errors)


def validate_refined_keypoint_publication(
    manifest: Mapping[str, Any],
    *,
    arrays: Mapping[str, Any],
    source_crop_arrays: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    quality_manifest: Mapping[str, Any],
    crop_manifest: Mapping[str, Any],
    raw_arrays: Mapping[str, Any],
    quality_arrays: Mapping[str, Any],
    retired_instance_keys: Sequence[int],
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    parent_manifest: Mapping[str, Any] | None = None,
    parent_arrays: Mapping[str, Any] | None = None,
    parent_retired_instance_keys: Sequence[int] | None = None,
) -> tuple[str, ...]:
    errors = list(validate_refined_keypoint_run_manifest(manifest))
    parsed = _parse_manifest_components(manifest)
    _, payload, dimensions, source, identity, review, reason = parsed
    if any(
        value is None
        for value in (payload, dimensions, source, identity, review, reason)
    ):
        return (*errors, "refined-keypoint manifest components are invalid")
    if any(
        value is not None
        for value in (parent_manifest, parent_arrays, parent_retired_instance_keys)
    ):
        errors.append("successor publication is deferred to the delta compactor")
    try:
        observed_source = build_refined_keypoint_source_bindings(
            raw_manifest=raw_manifest,
            quality_manifest=quality_manifest,
            crop_manifest=crop_manifest,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"refined-keypoint source validation failed: {exc}")
    else:
        if observed_source != source:
            errors.append("refined-keypoint source bindings mismatch")
    errors.extend(
        validate_refined_keypoint_snapshot_identity(
            identity,
            arrays=arrays,
            raw_arrays=raw_arrays,
            retired_instance_keys=retired_instance_keys,
        )
    )
    try:
        content = refined_keypoint_logical_content_document(
            arrays,
            dimensions=dimensions,
            source=source,
            identity=identity,
            source_crop_arrays=source_crop_arrays,
            review_state_map=review,
            reason_code_map=reason,
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"refined-keypoint logical array validation failed: {exc}")
    else:
        if content != payload["logical_content"]["document"]:
            errors.append("refined-keypoint logical_content differs from arrays")
    preserved_paths = (
        "instance_key",
        "source_crop_row_ids",
        "source_acquisition_frame_index",
        "frame_indices",
        "frame_row_offsets",
        "source_crop_row_signature",
        "pose_confidence",
        "pose_bbox_xyxy_roi",
        "pose_bbox_xyxy_img",
    )
    for path in preserved_paths:
        if (
            path in arrays
            and path in raw_arrays
            and not np.array_equal(
                _array_values(arrays[path]),
                _array_values(raw_arrays[path]),
                equal_nan=True,
            )
        ):
            errors.append(f"refined-keypoint source fact differs at {path}")
    if (
        "source_success" in arrays
        and "pose_success" in raw_arrays
        and not np.array_equal(
            _array_values(arrays["source_success"]),
            _array_values(raw_arrays["pose_success"]),
        )
    ):
        errors.append("refined source_success differs from raw pose_success")
    for path in ("instance_key", "frame_indices", "frame_row_offsets"):
        if (
            path in quality_arrays
            and path in raw_arrays
            and not np.array_equal(
                _array_values(quality_arrays[path]), _array_values(raw_arrays[path])
            )
        ):
            errors.append(f"quality/raw row identity differs at {path}")
    try:
        metadata_digest = refined_keypoint_metadata_declarations_digest(
            direct_metadata_declarations,
            consolidated_by_path=consolidated_metadata_declarations,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"refined-keypoint metadata validation failed: {exc}")
    else:
        if metadata_digest != payload["publication"].get(
            "metadata_declarations_digest"
        ):
            errors.append("refined-keypoint metadata digest mismatch")
    storage = payload.get("storage_plan")
    raw_profile = (
        storage.get("storage_profile") if isinstance(storage, Mapping) else None
    )
    try:
        if not isinstance(raw_profile, Mapping):
            raise ValueError("refined-keypoint storage profile is missing")
        profile = storage_profile_from_manifest(raw_profile)
        plans = plan_refined_keypoint_storage(dimensions, profile=profile)
    except (TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct refined physical plan: {exc}")
    else:
        bindings = {
            binding.path: binding for binding in REFINED_KEYPOINT_SCHEMA_V2.bindings
        }
        for entry in plans.entries:
            declaration = direct_metadata_declarations.get(entry.rule.path)
            if not isinstance(declaration, Mapping):
                errors.append(f"missing direct metadata at {entry.rule.path}")
                continue
            binding = bindings[entry.rule.path]
            contract = REFINED_KEYPOINT_SCHEMA_V2.contracts.resolve(
                binding.contract_id, binding.contract_version
            )
            errors.extend(
                f"refined physical metadata at {entry.rule.path}: {error}"
                for error in validate_array_metadata_declaration_from_plan(
                    declaration,
                    contract=contract,
                    plan=entry.plan,
                    fill_value=0,
                )
            )
    return tuple(errors)


__all__ = [
    "REFINED_KEYPOINT_ARRAY_DIGEST_ALGORITHM",
    "REFINED_KEYPOINT_LOGICAL_CONTENT_SCHEMA_ID",
    "REFINED_KEYPOINT_METADATA_DIGEST_SCOPE",
    "REFINED_KEYPOINT_RUN_MANIFEST_ATTRIBUTE",
    "REFINED_KEYPOINT_RUN_MANIFEST_PERSISTED_PATH",
    "REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_ID",
    "REFINED_KEYPOINT_RUN_MANIFEST_SCHEMA_VERSION",
    "RefinedKeypointSnapshotIdentity",
    "RefinedKeypointSourceBindings",
    "build_refined_keypoint_run_manifest",
    "build_refined_keypoint_source_bindings",
    "initial_refined_keypoint_snapshot_identity",
    "normalized_retired_instance_keys",
    "refined_keypoint_code_maps_from_manifest",
    "refined_keypoint_logical_content_document",
    "refined_keypoint_metadata_declarations_digest",
    "refined_keypoint_snapshot_identity_from_manifest",
    "refined_keypoint_source_bindings_from_manifest",
    "retired_instance_keys_digest",
    "validate_refined_keypoint_publication",
    "validate_refined_keypoint_run_manifest",
    "validate_refined_keypoint_snapshot_identity",
]
