"""Strict persisted manifest and publication gate for subject-mask quality v1."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_factory import (
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr.subject_mask_quality_producer import (
    SUBJECT_MASK_QUALITY_POLICY_SCHEMA_ID,
    SUBJECT_MASK_QUALITY_POLICY_SCHEMA_VERSION,
    SUBJECT_V1_LR_QUALITY_PROFILE_ID,
    SubjectV1LrObservationQualityPolicy,
    quality_profile_for_policy,
)
from fisheye.shared.zarr.subject_mask_quality_schema import (
    SUBJECT_MASK_QUALITY_SCHEMA_V1,
    SubjectMaskQualityDimensions,
    SubjectMaskQualityMetricDefinition,
    SubjectMaskQualityProfile,
    SubjectMaskQualitySourceReference,
)
from fisheye.shared.zarr.subject_mask_quality_storage import (
    SubjectMaskQualityStoragePlanSet,
    plan_subject_mask_quality_storage,
)
from fisheye.shared.zarr.subject_mask_schema import SubjectMaskComponentRegistry
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
)

SUBJECT_MASK_QUALITY_RUN_MANIFEST_SCHEMA_ID = (
    "palette.subject_mask_quality.run_manifest"
)
SUBJECT_MASK_QUALITY_RUN_MANIFEST_SCHEMA_VERSION = 2
SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
SUBJECT_MASK_QUALITY_RUN_MANIFEST_PERSISTED_PATH = (
    "subject_mask_quality_runs/<run>/zarr.json.attributes.run_manifest"
)
SUBJECT_MASK_QUALITY_LOGICAL_CONTENT_SCHEMA_ID = (
    "palette.subject_mask_quality.logical_content"
)
SUBJECT_MASK_QUALITY_LOGICAL_CONTENT_SCHEMA_VERSION = 1
SUBJECT_MASK_QUALITY_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_bytes_v1"
SUBJECT_MASK_QUALITY_METADATA_DIGEST_SCOPE = (
    "exact_group_and_array_declarations_redacting_manifest_lifecycle_"
    "and_transport_publication_attrs"
)
_TRANSPORT_PUBLICATION_ATTRS = (
    "atomic_publication_owner_uuid",
    "atomic_publication_tombstone",
    "cluster_output_staging",
    "publication_status",
    "subject_mask_bundle_selector_eligible",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
    return normalized


def _require_run_id(value: object) -> str:
    normalized = str(value).strip()
    if not normalized or "/" in normalized:
        raise ValueError("run_id must be one nonempty archive group name.")
    return normalized


def _array_shape_dtype(value: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    try:
        shape = tuple(int(item) for item in value.shape)
        dtype = np.dtype(value.dtype)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("Array lacks exact shape or dtype metadata.") from exc
    return shape, dtype


def streaming_array_sha256(value: Any, *, row_block_rows: int = 65_536) -> str:
    """Hash C-order decoded bytes without materializing a complete array."""

    shape, _dtype = _array_shape_dtype(value)
    block_rows = int(row_block_rows)
    if block_rows <= 0:
        raise ValueError("row_block_rows must be positive.")
    digest = hashlib.sha256()
    if not shape:
        block = np.ascontiguousarray(np.asarray(value[...]))
        digest.update(block.view(np.uint8))
        return digest.hexdigest()
    trailing = (slice(None),) * (len(shape) - 1)
    for start in range(0, shape[0], block_rows):
        stop = min(start + block_rows, shape[0])
        selection = (slice(start, stop), *trailing)
        block = np.ascontiguousarray(np.asarray(value[selection]))
        digest.update(block.view(np.uint8))
    return digest.hexdigest()


def _metric_from_manifest(
    value: Mapping[str, Any],
) -> SubjectMaskQualityMetricDefinition:
    expected = {
        "metric_id",
        "metric_version",
        "units",
        "higher_is_worse",
        "description",
    }
    if set(value) != expected:
        raise ValueError("Subject-mask quality metric has an unexpected field set.")
    metric = SubjectMaskQualityMetricDefinition(
        metric_id=value.get("metric_id"),
        metric_version=value.get("metric_version"),
        units=value.get("units"),
        higher_is_worse=value.get("higher_is_worse"),
        description=value.get("description"),
    )
    if dict(value) != metric.as_manifest():
        raise ValueError("Subject-mask quality metric is not canonical.")
    return metric


def _flag_map_from_manifest(value: object, *, name: str) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object.")
    result: dict[int, str] = {}
    for raw_key, raw_label in value.items():
        key = str(raw_key)
        try:
            bit = int(key)
        except ValueError as exc:
            raise ValueError(f"{name} keys must be canonical decimal bits.") from exc
        if str(bit) != key:
            raise ValueError(f"{name} keys must be canonical decimal bits.")
        result[bit] = str(raw_label)
    return result


def quality_profile_from_manifest(
    value: Mapping[str, Any],
) -> SubjectMaskQualityProfile:
    expected = {
        "profile_id",
        "profile_version",
        "policy_digest",
        "component_metrics",
        "observation_metrics",
        "component_flag_map",
        "observation_flag_map",
        "zero_flag_semantics",
        "profile_digest",
    }
    if set(value) != expected:
        raise ValueError("Subject-mask quality profile has an unexpected field set.")
    component_metrics = value.get("component_metrics")
    observation_metrics = value.get("observation_metrics")
    if not isinstance(component_metrics, list) or not isinstance(
        observation_metrics, list
    ):
        raise TypeError("Subject-mask quality metric catalogs must be arrays.")
    if value.get("zero_flag_semantics") != "no_quality_finding":
        raise ValueError("Subject-mask quality zero-flag semantics mismatch.")
    profile = SubjectMaskQualityProfile(
        profile_id=value.get("profile_id"),
        profile_version=value.get("profile_version"),
        policy_digest=value.get("policy_digest"),
        component_metrics=tuple(
            _metric_from_manifest(item) for item in component_metrics
        ),
        observation_metrics=tuple(
            _metric_from_manifest(item) for item in observation_metrics
        ),
        component_flag_map=_flag_map_from_manifest(
            value.get("component_flag_map"), name="component_flag_map"
        ),
        observation_flag_map=_flag_map_from_manifest(
            value.get("observation_flag_map"), name="observation_flag_map"
        ),
    )
    if dict(value) != profile.as_manifest():
        raise ValueError("Subject-mask quality profile is not canonical.")
    return profile


def quality_source_from_manifest(
    value: Mapping[str, Any],
) -> SubjectMaskQualitySourceReference:
    expected = {
        "stage",
        "run_name",
        "run_path",
        "schema_id",
        "schema_version",
        "manifest_digest",
        "dense_array_values_sha256",
        "component_registry_digest",
        "source_array_values_sha256",
        "coverage",
    }
    if set(value) != expected:
        raise ValueError("Subject-mask quality source has an unexpected field set.")
    source = SubjectMaskQualitySourceReference(
        run_name=value.get("run_name"),
        manifest_digest=value.get("manifest_digest"),
        dense_array_values_sha256=value.get("dense_array_values_sha256"),
        component_registry_digest=value.get("component_registry_digest"),
        source_array_values_sha256=value.get("source_array_values_sha256"),
    )
    if dict(value) != source.as_manifest():
        raise ValueError("Subject-mask quality source is not canonical.")
    return source


def quality_policy_from_manifest(
    value: Mapping[str, Any],
) -> SubjectV1LrObservationQualityPolicy:
    expected = {
        "schema_id",
        "schema_version",
        "policy_id",
        "required_components",
        "component_semantics",
        "allowed_overlap",
        "exclusive_pairs",
        "maximum_outside_body_fraction",
        "maximum_exclusive_pair_overlap_fraction",
        "relation_fraction_denominators",
        "component_proposal_unusable_flags",
        "observation_proposal_unusable_flags",
        "automatic_pixel_mutation",
        "accepted_review_state_ownership",
        "temporal_metrics",
        "policy_digest",
    }
    if set(value) != expected:
        raise ValueError("Subject-mask quality policy has an unexpected field set.")
    if (
        value.get("schema_id") != SUBJECT_MASK_QUALITY_POLICY_SCHEMA_ID
        or value.get("schema_version") != SUBJECT_MASK_QUALITY_POLICY_SCHEMA_VERSION
        or value.get("policy_id") != SUBJECT_V1_LR_QUALITY_PROFILE_ID
    ):
        raise ValueError("Subject-mask quality policy identity mismatch.")
    for name in (
        "maximum_outside_body_fraction",
        "maximum_exclusive_pair_overlap_fraction",
    ):
        if type(value.get(name)) is not float:
            raise TypeError(f"Subject-mask quality policy {name} must be a float.")
    policy = SubjectV1LrObservationQualityPolicy(
        maximum_outside_body_fraction=value.get("maximum_outside_body_fraction"),
        maximum_exclusive_pair_overlap_fraction=value.get(
            "maximum_exclusive_pair_overlap_fraction"
        ),
        policy_version=value.get("schema_version"),
    )
    if dict(value) != policy.as_manifest():
        raise ValueError("Subject-mask quality policy differs from its builder.")
    return policy


def component_registry_from_manifest(
    value: Mapping[str, Any],
) -> SubjectMaskComponentRegistry:
    expected = {
        "schema_id",
        "schema_version",
        "labels",
        "channel_axis",
        "ordering",
    }
    if set(value) != expected or not isinstance(value.get("labels"), list):
        raise ValueError("Subject-mask component registry is not exact.")
    registry = SubjectMaskComponentRegistry(tuple(value["labels"]))
    if dict(value) != registry.as_manifest():
        raise ValueError("Subject-mask component registry is not canonical.")
    return registry


def _dimensions_from_logical(
    logical: Mapping[str, Any],
) -> SubjectMaskQualityDimensions:
    raw = logical.get("dimensions")
    expected = {
        "n_frames",
        "n_frame_boundaries",
        "n_instances",
        "n_rois",
        "n_channels",
        "H",
        "W",
        "n_component_metrics",
        "n_observation_metrics",
        "roi_height",
        "roi_width",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("Subject-mask quality dimensions are not exact.")
    dimensions = SubjectMaskQualityDimensions(
        n_frames=raw.get("n_frames"),
        n_rois=raw.get("n_rois"),
        n_channels=raw.get("n_channels"),
        roi_height=raw.get("roi_height"),
        roi_width=raw.get("roi_width"),
        n_component_metrics=raw.get("n_component_metrics"),
        n_observation_metrics=raw.get("n_observation_metrics"),
    )
    if dict(raw) != dimensions.as_manifest():
        raise ValueError("Subject-mask quality dimensions are not canonical.")
    return dimensions


def subject_mask_quality_logical_content_document(
    arrays: Mapping[str, Any],
    *,
    dimensions: SubjectMaskQualityDimensions,
    components: SubjectMaskComponentRegistry,
    profile: SubjectMaskQualityProfile,
    source: SubjectMaskQualitySourceReference,
    source_arrays: Mapping[str, Any] | None = None,
    validate_logical_arrays: bool,
    digest_block_rows: int = 65_536,
) -> dict[str, object]:
    if validate_logical_arrays:
        SUBJECT_MASK_QUALITY_SCHEMA_V1.require(
            arrays,
            dimensions=dimensions,
            components=components,
            profile=profile,
            source_mask_arrays=source_arrays,
        )
    declarations: dict[str, object] = {}
    for path in SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths:
        shape, dtype = _array_shape_dtype(arrays[path])
        declarations[path] = {
            "shape": list(shape),
            "dtype": str(dtype),
            "digest_algorithm": SUBJECT_MASK_QUALITY_ARRAY_DIGEST_ALGORITHM,
            "sha256": streaming_array_sha256(
                arrays[path], row_block_rows=digest_block_rows
            ),
        }
    return {
        "schema_id": SUBJECT_MASK_QUALITY_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_QUALITY_LOGICAL_CONTENT_SCHEMA_VERSION,
        "logical_schema": {
            "id": SUBJECT_MASK_QUALITY_SCHEMA_V1.schema_id,
            "version": SUBJECT_MASK_QUALITY_SCHEMA_V1.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "component_registry_digest": source.component_registry_digest,
        "source_manifest_digest": source.manifest_digest,
        "source_dense_array_values_sha256": source.dense_array_values_sha256,
        "source_array_values_sha256": dict(source.source_array_values_sha256),
        "profile_digest": profile.profile_digest,
        "policy_digest": profile.policy_digest,
        "arrays": declarations,
    }


def subject_mask_quality_metadata_declarations_digest(
    direct_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_by_path: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = {"", *SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths}
    if set(direct_by_path) != expected or set(consolidated_by_path) != expected:
        raise ValueError(
            "Subject-mask quality metadata declaration paths are incomplete."
        )
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
                    raise ValueError("Quality run group attributes must be an object.")
                redacted = dict(attributes)
                for name in (
                    SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE,
                    "status",
                    RUN_COMPLETION_STATUS_ATTR,
                    RUN_COMPLETED_AT_ATTR,
                    *_TRANSPORT_PUBLICATION_ATTRS,
                ):
                    redacted.pop(name, None)
                declaration["attributes"] = redacted
        if direct != consolidated:
            raise ValueError(
                f"Direct and consolidated metadata differ at {path or '<run>'!r}."
            )
        normalized[path] = direct
    return canonical_json_sha256(
        {
            "scope": SUBJECT_MASK_QUALITY_METADATA_DIGEST_SCOPE,
            "declarations": normalized,
        }
    )


def subject_mask_quality_output_write_units(
    storage_plan: SubjectMaskQualityStoragePlanSet,
) -> dict[str, object]:
    """Return exact complete physical units used by the final publisher."""

    result: dict[str, object] = {}
    for entry in storage_plan.entries:
        plan = entry.plan
        unit_shape = plan.shard_shape or plan.chunk_shape
        if unit_shape is None:
            raise ValueError("Subject-mask quality does not support scalar arrays.")
        result[entry.rule.path] = {
            "kind": "outer_shard" if plan.shard_shape is not None else "inner_chunk",
            "shape": list(unit_shape),
            "row_count": int(unit_shape[0]),
            "write_ownership": plan.write_ownership,
        }
    return result


def build_subject_mask_quality_run_manifest(
    *,
    run_id: str,
    dimensions: SubjectMaskQualityDimensions,
    components: SubjectMaskComponentRegistry,
    profile: SubjectMaskQualityProfile,
    policy: SubjectV1LrObservationQualityPolicy,
    source: SubjectMaskQualitySourceReference,
    source_manifest: Mapping[str, Any],
    storage_plan: SubjectMaskQualityStoragePlanSet,
    arrays: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    write_receipt: Mapping[str, Any],
) -> dict[str, object]:
    """Build the exact selector-ineligible persisted manifest envelope."""

    resolved_run_id = _require_run_id(run_id)
    if canonical_json_sha256(source_manifest) != source.manifest_digest:
        raise ValueError("Source mask manifest differs from the bound digest.")
    if profile.policy_digest != policy.policy_digest:
        raise ValueError("Quality profile and policy digests differ.")
    if profile.as_manifest() != quality_profile_for_policy(policy).as_manifest():
        raise ValueError("Quality profile differs from the frozen policy builder.")
    if storage_plan.dimensions != dimensions:
        raise ValueError("Quality storage-plan dimensions differ.")
    content = subject_mask_quality_logical_content_document(
        arrays,
        dimensions=dimensions,
        components=components,
        profile=profile,
        source=source,
        source_arrays=source_arrays,
        validate_logical_arrays=True,
    )
    metadata_digest = subject_mask_quality_metadata_declarations_digest(
        direct_metadata_declarations,
        consolidated_by_path=consolidated_metadata_declarations,
    )
    payload: dict[str, object] = {
        "run_id": resolved_run_id,
        "stage": "subject_mask_quality",
        "publication": {
            "artifact_class": "observation_local_quality_diagnostics",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": (
                SUBJECT_MASK_QUALITY_METADATA_DIGEST_SCOPE
            ),
            "metadata_declarations_digest_algorithm": (CANONICAL_JSON_DIGEST_ALGORITHM),
            "metadata_declarations_digest": metadata_digest,
        },
        "logical_schema": SUBJECT_MASK_QUALITY_SCHEMA_V1.as_manifest(
            dimensions=dimensions,
            components=components,
            profile=profile,
            source=source,
        ),
        "storage_plan": storage_plan.as_manifest(),
        "source_refined_subject_mask_snapshot": source.as_manifest(),
        "policy": policy.as_manifest(),
        "write_receipt": dict(write_receipt),
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(content),
            "document": content,
        },
    }
    envelope = {
        "schema_id": SUBJECT_MASK_QUALITY_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": SUBJECT_MASK_QUALITY_RUN_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE,
        "persisted_path": SUBJECT_MASK_QUALITY_RUN_MANIFEST_PERSISTED_PATH,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def _parse_manifest_components(
    manifest: Mapping[str, Any],
) -> tuple[
    list[str],
    Mapping[str, Any] | None,
    SubjectMaskQualityDimensions | None,
    SubjectMaskComponentRegistry | None,
    SubjectMaskQualityProfile | None,
    SubjectMaskQualitySourceReference | None,
    SubjectV1LrObservationQualityPolicy | None,
]:
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
        errors.append("subject-mask quality manifest envelope has unexpected fields")
    if (
        manifest.get("schema_id") != SUBJECT_MASK_QUALITY_RUN_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version")
        != SUBJECT_MASK_QUALITY_RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("persisted_attribute")
        != SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE
        or manifest.get("persisted_path")
        != SUBJECT_MASK_QUALITY_RUN_MANIFEST_PERSISTED_PATH
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("subject-mask quality manifest envelope identity mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (
            [*errors, "subject-mask quality manifest payload must be an object"],
            None,
            None,
            None,
            None,
            None,
            None,
        )
    try:
        expected_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"subject-mask quality manifest is not strict JSON: {exc}")
    else:
        if manifest.get("payload_digest") != expected_digest:
            errors.append("subject-mask quality manifest payload_digest mismatch")
    expected_payload = {
        "run_id",
        "stage",
        "publication",
        "logical_schema",
        "storage_plan",
        "source_refined_subject_mask_snapshot",
        "policy",
        "write_receipt",
        "logical_content",
    }
    if set(payload) != expected_payload:
        errors.append("subject-mask quality manifest payload has unexpected fields")
    try:
        _require_run_id(payload.get("run_id"))
    except ValueError as exc:
        errors.append(str(exc))
    if payload.get("stage") != "subject_mask_quality":
        errors.append("subject-mask quality manifest stage mismatch")

    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("subject-mask quality publication must be an object")
    else:
        expected_publication = {
            "artifact_class": "observation_local_quality_diagnostics",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": (
                SUBJECT_MASK_QUALITY_METADATA_DIGEST_SCOPE
            ),
            "metadata_declarations_digest_algorithm": (CANONICAL_JSON_DIGEST_ALGORITHM),
            "metadata_declarations_digest": publication.get(
                "metadata_declarations_digest"
            ),
        }
        if dict(publication) != expected_publication:
            errors.append("subject-mask quality publication is not exact")
        try:
            _require_sha256(
                publication.get("metadata_declarations_digest"),
                name="metadata_declarations_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))

    dimensions = None
    components = None
    profile = None
    source = None
    logical = payload.get("logical_schema")
    if not isinstance(logical, Mapping):
        errors.append("subject-mask quality logical_schema must be an object")
    else:
        try:
            dimensions = _dimensions_from_logical(logical)
            component_value = logical.get("components")
            profile_value = logical.get("profile")
            source_value = logical.get("source")
            if not all(
                isinstance(item, Mapping)
                for item in (component_value, profile_value, source_value)
            ):
                raise TypeError(
                    "Quality logical components/profile/source must be objects."
                )
            components = component_registry_from_manifest(component_value)
            profile = quality_profile_from_manifest(profile_value)
            source = quality_source_from_manifest(source_value)
            expected_logical = SUBJECT_MASK_QUALITY_SCHEMA_V1.as_manifest(
                dimensions=dimensions,
                components=components,
                profile=profile,
                source=source,
            )
            if dict(logical) != expected_logical:
                errors.append(
                    "subject-mask quality logical_schema differs from builder"
                )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    source_value = payload.get("source_refined_subject_mask_snapshot")
    if isinstance(source_value, Mapping):
        try:
            top_source = quality_source_from_manifest(source_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if source is not None and top_source != source:
                errors.append("subject-mask quality source declarations disagree")
            source = top_source
    else:
        errors.append("source_refined_subject_mask_snapshot must be an object")

    policy = None
    policy_value = payload.get("policy")
    if isinstance(policy_value, Mapping):
        try:
            policy = quality_policy_from_manifest(policy_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    else:
        errors.append("subject-mask quality policy must be an object")
    if policy is not None and profile is not None:
        if policy.policy_digest != profile.policy_digest:
            errors.append("subject-mask quality policy/profile digests differ")
        if profile.as_manifest() != quality_profile_for_policy(policy).as_manifest():
            errors.append("subject-mask quality profile differs from policy builder")
    if components is not None and source is not None:
        if source.component_registry_digest != canonical_json_sha256(
            components.as_manifest()
        ):
            errors.append("subject-mask quality component registry digest mismatch")
    return errors, payload, dimensions, components, profile, source, policy


def _validate_logical_content(
    payload: Mapping[str, Any],
    *,
    dimensions: SubjectMaskQualityDimensions | None,
    profile: SubjectMaskQualityProfile | None,
    source: SubjectMaskQualitySourceReference | None,
    errors: list[str],
) -> None:
    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping) or set(logical_content) != {
        "digest_algorithm",
        "digest",
        "document",
    }:
        errors.append("subject-mask quality logical_content envelope is invalid")
        return
    document = logical_content.get("document")
    if logical_content.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("subject-mask quality logical_content algorithm mismatch")
    if not isinstance(document, Mapping):
        errors.append("subject-mask quality logical_content document must be an object")
        return
    try:
        digest = canonical_json_sha256(document)
    except (TypeError, ValueError) as exc:
        errors.append(f"subject-mask quality logical_content is not strict JSON: {exc}")
    else:
        if logical_content.get("digest") != digest:
            errors.append("subject-mask quality logical_content digest mismatch")
    expected_fields = {
        "schema_id",
        "schema_version",
        "logical_schema",
        "dimensions",
        "component_registry_digest",
        "source_manifest_digest",
        "source_dense_array_values_sha256",
        "source_array_values_sha256",
        "profile_digest",
        "policy_digest",
        "arrays",
    }
    if set(document) != expected_fields:
        errors.append("subject-mask quality logical_content has unexpected fields")
    if (
        document.get("schema_id") != SUBJECT_MASK_QUALITY_LOGICAL_CONTENT_SCHEMA_ID
        or document.get("schema_version")
        != SUBJECT_MASK_QUALITY_LOGICAL_CONTENT_SCHEMA_VERSION
        or document.get("logical_schema")
        != {
            "id": SUBJECT_MASK_QUALITY_SCHEMA_V1.schema_id,
            "version": SUBJECT_MASK_QUALITY_SCHEMA_V1.schema_version,
        }
    ):
        errors.append("subject-mask quality logical_content identity mismatch")
    if (
        dimensions is not None
        and document.get("dimensions") != dimensions.as_manifest()
    ):
        errors.append("subject-mask quality logical_content dimensions mismatch")
    if source is not None:
        expected_source = {
            "component_registry_digest": source.component_registry_digest,
            "source_manifest_digest": source.manifest_digest,
            "source_dense_array_values_sha256": source.dense_array_values_sha256,
            "source_array_values_sha256": dict(source.source_array_values_sha256),
        }
        for name, expected_value in expected_source.items():
            if document.get(name) != expected_value:
                errors.append(f"subject-mask quality logical_content {name} mismatch")
    if profile is not None:
        if document.get("profile_digest") != profile.profile_digest:
            errors.append("subject-mask quality logical_content profile mismatch")
        if document.get("policy_digest") != profile.policy_digest:
            errors.append("subject-mask quality logical_content policy mismatch")
    array_docs = document.get("arrays")
    if not isinstance(array_docs, Mapping) or set(array_docs) != set(
        SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths
    ):
        errors.append("subject-mask quality array declarations mismatch")
        return
    bindings = {
        binding.path: binding for binding in SUBJECT_MASK_QUALITY_SCHEMA_V1.bindings
    }
    for path, item in array_docs.items():
        if not isinstance(item, Mapping) or set(item) != {
            "shape",
            "dtype",
            "digest_algorithm",
            "sha256",
        }:
            errors.append(f"subject-mask quality declaration invalid at {path}")
            continue
        if dimensions is not None:
            binding = bindings[path]
            contract = SUBJECT_MASK_QUALITY_SCHEMA_V1.contracts.resolve(
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
                errors.append(f"subject-mask quality shape mismatch at {path}")
            if item.get("dtype") != str(contract.dtype.numpy_dtype):
                errors.append(f"subject-mask quality dtype mismatch at {path}")
        if item.get("digest_algorithm") != SUBJECT_MASK_QUALITY_ARRAY_DIGEST_ALGORITHM:
            errors.append(f"subject-mask quality digest algorithm mismatch at {path}")
        try:
            _require_sha256(item.get("sha256"), name=f"{path} sha256")
        except ValueError as exc:
            errors.append(str(exc))


def validate_subject_mask_quality_run_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Deeply validate the persisted envelope without array reads."""

    errors, payload, dimensions, _components, profile, source, policy = (
        _parse_manifest_components(manifest)
    )
    if payload is None:
        return tuple(errors)
    storage = payload.get("storage_plan")
    if not isinstance(storage, Mapping):
        errors.append("subject-mask quality storage_plan must be an object")
    elif dimensions is not None:
        raw_profile = storage.get("storage_profile")
        if not isinstance(raw_profile, Mapping):
            errors.append("subject-mask quality storage profile must be an object")
        else:
            try:
                storage_profile = storage_profile_from_manifest(raw_profile)
                reconstructed_plan = plan_subject_mask_quality_storage(
                    dimensions, profile=storage_profile
                )
                expected_storage = reconstructed_plan.as_manifest()
            except (TypeError, ValueError) as exc:
                errors.append(f"cannot reconstruct subject-mask quality storage: {exc}")
            else:
                if dict(storage) != expected_storage:
                    errors.append(
                        "subject-mask quality storage plan differs from planner output"
                    )
                receipt = payload.get("write_receipt")
                if isinstance(receipt, Mapping) and receipt.get(
                    "output_array_write_units"
                ) != subject_mask_quality_output_write_units(reconstructed_plan):
                    errors.append(
                        "subject-mask quality output write units differ from storage plan"
                    )
    receipt = payload.get("write_receipt")
    expected_receipt_fields = {
        "schema_id",
        "schema_version",
        "source_compute_block_rows",
        "source_compute_block_bytes_budget",
        "source_compute_block_count",
        "output_write_unit",
        "output_array_write_units",
        "scratch_surface",
        "parallel_write_policy",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_receipt_fields:
        errors.append("subject-mask quality write_receipt is not exact")
    else:
        if (
            receipt.get("schema_id") != "palette.subject_mask_quality.write_receipt"
            or receipt.get("schema_version") != 1
            or receipt.get("output_write_unit")
            != "complete_outer_shard_or_unsharded_chunk"
            or receipt.get("scratch_surface")
            != "node_local_npy_memmap_deleted_after_publication"
            or receipt.get("parallel_write_policy")
            != "single_writer_v1_future_workers_require_disjoint_whole_shards"
        ):
            errors.append("subject-mask quality write_receipt identity mismatch")
        for name in (
            "source_compute_block_rows",
            "source_compute_block_bytes_budget",
            "source_compute_block_count",
        ):
            if type(receipt.get(name)) is not int or int(receipt[name]) <= 0:
                errors.append(f"subject-mask quality write_receipt {name} is invalid")
        if dimensions is not None and all(
            type(receipt.get(name)) is int
            for name in (
                "source_compute_block_rows",
                "source_compute_block_bytes_budget",
                "source_compute_block_count",
            )
        ):
            row_bytes = (
                dimensions.n_channels * dimensions.roi_height * dimensions.roi_width
            )
            expected_rows = max(
                1,
                int(receipt["source_compute_block_bytes_budget"]) // max(1, row_bytes),
            )
            expected_count = (dimensions.n_rois + expected_rows - 1) // expected_rows
            if receipt.get("source_compute_block_rows") != expected_rows:
                errors.append(
                    "subject-mask quality effective compute block rows mismatch"
                )
            if receipt.get("source_compute_block_count") != expected_count:
                errors.append("subject-mask quality compute block count mismatch")
        units = receipt.get("output_array_write_units")
        if not isinstance(units, Mapping) or set(units) != set(
            SUBJECT_MASK_QUALITY_SCHEMA_V1.binding_paths
        ):
            errors.append("subject-mask quality output write units are incomplete")
    _validate_logical_content(
        payload,
        dimensions=dimensions,
        profile=profile,
        source=source,
        errors=errors,
    )
    if policy is None:
        errors.append("subject-mask quality policy could not be reconstructed")
    return tuple(errors)


def validate_subject_mask_quality_publication(
    manifest: Mapping[str, Any],
    *,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recompute decoded, source, metadata, and physical evidence."""

    errors = list(validate_subject_mask_quality_run_manifest(manifest))
    (
        _,
        payload,
        dimensions,
        components,
        profile,
        source,
        _,
    ) = _parse_manifest_components(manifest)
    if any(item is None for item in (payload, dimensions, components, profile, source)):
        return (*errors, "subject-mask quality manifest components are invalid")
    try:
        source_manifest_digest = canonical_json_sha256(source_manifest)
    except (TypeError, ValueError) as exc:
        errors.append(f"subject-mask quality source manifest is not strict JSON: {exc}")
    else:
        if source_manifest_digest != source.manifest_digest:
            errors.append("subject-mask quality source manifest digest mismatch")
    try:
        content = subject_mask_quality_logical_content_document(
            arrays,
            dimensions=dimensions,
            components=components,
            profile=profile,
            source=source,
            validate_logical_arrays=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"subject-mask quality array digest failed: {exc}")
    else:
        if content != payload["logical_content"]["document"]:
            errors.append("subject-mask quality content differs from decoded arrays")
    try:
        metadata_digest = subject_mask_quality_metadata_declarations_digest(
            direct_metadata_declarations,
            consolidated_by_path=consolidated_metadata_declarations,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"subject-mask quality metadata validation failed: {exc}")
    else:
        if metadata_digest != payload["publication"].get(
            "metadata_declarations_digest"
        ):
            errors.append("subject-mask quality metadata digest mismatch")
    storage = payload.get("storage_plan")
    raw_profile = (
        storage.get("storage_profile") if isinstance(storage, Mapping) else None
    )
    try:
        if not isinstance(raw_profile, Mapping):
            raise ValueError("subject-mask quality storage profile is missing")
        storage_profile = storage_profile_from_manifest(raw_profile)
        plans = plan_subject_mask_quality_storage(dimensions, profile=storage_profile)
    except (TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct subject-mask quality physical plan: {exc}")
    else:
        bindings = {
            binding.path: binding for binding in SUBJECT_MASK_QUALITY_SCHEMA_V1.bindings
        }
        for entry in plans.entries:
            declaration = direct_metadata_declarations.get(entry.rule.path)
            if not isinstance(declaration, Mapping):
                errors.append(f"missing direct metadata at {entry.rule.path}")
                continue
            binding = bindings[entry.rule.path]
            contract = SUBJECT_MASK_QUALITY_SCHEMA_V1.contracts.resolve(
                binding.contract_id, binding.contract_version
            )
            errors.extend(
                f"subject-mask quality physical metadata at {entry.rule.path}: {error}"
                for error in validate_array_metadata_declaration_from_plan(
                    declaration,
                    contract=contract,
                    plan=entry.plan,
                    fill_value=0,
                )
            )
    return tuple(errors)


__all__ = [
    "SUBJECT_MASK_QUALITY_ARRAY_DIGEST_ALGORITHM",
    "SUBJECT_MASK_QUALITY_LOGICAL_CONTENT_SCHEMA_ID",
    "SUBJECT_MASK_QUALITY_LOGICAL_CONTENT_SCHEMA_VERSION",
    "SUBJECT_MASK_QUALITY_METADATA_DIGEST_SCOPE",
    "SUBJECT_MASK_QUALITY_RUN_MANIFEST_ATTRIBUTE",
    "SUBJECT_MASK_QUALITY_RUN_MANIFEST_PERSISTED_PATH",
    "SUBJECT_MASK_QUALITY_RUN_MANIFEST_SCHEMA_ID",
    "SUBJECT_MASK_QUALITY_RUN_MANIFEST_SCHEMA_VERSION",
    "build_subject_mask_quality_run_manifest",
    "component_registry_from_manifest",
    "quality_policy_from_manifest",
    "quality_profile_from_manifest",
    "quality_source_from_manifest",
    "streaming_array_sha256",
    "subject_mask_quality_logical_content_document",
    "subject_mask_quality_metadata_declarations_digest",
    "subject_mask_quality_output_write_units",
    "validate_subject_mask_quality_publication",
    "validate_subject_mask_quality_run_manifest",
]
