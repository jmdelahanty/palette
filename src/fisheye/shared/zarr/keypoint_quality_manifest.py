"""Strict run manifest and publication gate for keypoint-quality v1."""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_factory import (
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.keypoint_quality_producer import (
    KEYPOINT_QUALITY_POLICY_SCHEMA_ID,
    KEYPOINT_QUALITY_POLICY_SCHEMA_VERSION,
    OBSERVATION_LOCAL_QUALITY_PROFILE_ID,
    ObservationLocalKeypointQualityPolicy,
    quality_profile_for_policy,
)
from fisheye.shared.zarr.keypoint_quality_schema import (
    KEYPOINT_QUALITY_SCHEMA_V1,
    KeypointQualityDimensions,
    KeypointQualityProfile,
    KeypointQualitySourceReference,
    QualityMetricDefinition,
)
from fisheye.shared.zarr.keypoint_quality_storage import (
    KeypointQualityStoragePlanSet,
    plan_keypoint_quality_storage,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest


KEYPOINT_QUALITY_RUN_MANIFEST_SCHEMA_ID = "palette.keypoint_quality.run_manifest"
KEYPOINT_QUALITY_RUN_MANIFEST_SCHEMA_VERSION = 1
KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
KEYPOINT_QUALITY_RUN_MANIFEST_PERSISTED_PATH = (
    "keypoint_quality_runs/<run>/zarr.json.attributes.run_manifest"
)
KEYPOINT_QUALITY_LOGICAL_CONTENT_SCHEMA_ID = (
    "palette.keypoint_quality.logical_content"
)
KEYPOINT_QUALITY_LOGICAL_CONTENT_SCHEMA_VERSION = 1
KEYPOINT_QUALITY_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_bytes_v1"
KEYPOINT_QUALITY_METADATA_DIGEST_SCOPE = (
    "exact_group_and_array_declarations_with_attributes_redacting_only_run_manifest"
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


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


def _metric_from_manifest(value: Mapping[str, Any]) -> QualityMetricDefinition:
    expected = {
        "metric_id",
        "metric_version",
        "units",
        "higher_is_worse",
        "description",
    }
    if set(value) != expected:
        raise ValueError("Quality metric definition has an unexpected field set.")
    metric = QualityMetricDefinition(
        metric_id=value.get("metric_id"),
        metric_version=value.get("metric_version"),
        units=value.get("units"),
        higher_is_worse=value.get("higher_is_worse"),
        description=value.get("description"),
    )
    if dict(value) != metric.as_manifest():
        raise ValueError("Quality metric definition is not canonical.")
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
) -> KeypointQualityProfile:
    expected = {
        "profile_id",
        "profile_version",
        "policy_digest",
        "keypoint_metrics",
        "pose_metrics",
        "keypoint_flag_map",
        "pose_flag_map",
        "zero_flag_semantics",
        "profile_digest",
    }
    if set(value) != expected:
        raise ValueError("Keypoint-quality profile has an unexpected field set.")
    raw_keypoint_metrics = value.get("keypoint_metrics")
    raw_pose_metrics = value.get("pose_metrics")
    if not isinstance(raw_keypoint_metrics, list) or not isinstance(
        raw_pose_metrics, list
    ):
        raise TypeError("Quality metric catalogs must be arrays.")
    if value.get("zero_flag_semantics") != "no_quality_finding":
        raise ValueError("Quality zero-flag semantics mismatch.")
    profile = KeypointQualityProfile(
        profile_id=value.get("profile_id"),
        profile_version=value.get("profile_version"),
        policy_digest=value.get("policy_digest"),
        keypoint_metrics=tuple(
            _metric_from_manifest(item) for item in raw_keypoint_metrics
        ),
        pose_metrics=tuple(_metric_from_manifest(item) for item in raw_pose_metrics),
        keypoint_flag_map=_flag_map_from_manifest(
            value.get("keypoint_flag_map"), name="keypoint_flag_map"
        ),
        pose_flag_map=_flag_map_from_manifest(
            value.get("pose_flag_map"), name="pose_flag_map"
        ),
    )
    if dict(value) != profile.as_manifest():
        raise ValueError("Keypoint-quality profile differs from its frozen builder.")
    return profile


def quality_source_from_manifest(
    value: Mapping[str, Any],
) -> KeypointQualitySourceReference:
    expected = {
        "stage",
        "run_name",
        "run_path",
        "schema_id",
        "schema_version",
        "manifest_digest",
        "skeleton_id",
        "skeleton_digest",
        "keypoint_row_signatures_digest",
        "coverage",
    }
    if set(value) != expected:
        raise ValueError("Keypoint-quality source has an unexpected field set.")
    source = KeypointQualitySourceReference(
        run_name=value.get("run_name"),
        manifest_digest=value.get("manifest_digest"),
        skeleton_id=value.get("skeleton_id"),
        skeleton_digest=value.get("skeleton_digest"),
        keypoint_row_signatures_digest=value.get(
            "keypoint_row_signatures_digest"
        ),
    )
    if dict(value) != source.as_manifest():
        raise ValueError("Keypoint-quality source differs from its frozen builder.")
    return source


def quality_policy_from_manifest(
    value: Mapping[str, Any],
) -> ObservationLocalKeypointQualityPolicy:
    expected = {
        "schema_id",
        "schema_version",
        "policy_id",
        "confidence_threshold",
        "minimum_valid_keypoints",
        "temporal_metrics",
        "heading_metrics",
        "policy_digest",
    }
    if set(value) != expected:
        raise ValueError("Keypoint-quality policy has an unexpected field set.")
    if (
        value.get("schema_id") != KEYPOINT_QUALITY_POLICY_SCHEMA_ID
        or value.get("schema_version") != KEYPOINT_QUALITY_POLICY_SCHEMA_VERSION
        or value.get("policy_id") != OBSERVATION_LOCAL_QUALITY_PROFILE_ID
        or value.get("temporal_metrics") != "forbidden"
        or value.get("heading_metrics") != "forbidden"
    ):
        raise ValueError("Keypoint-quality policy identity mismatch.")
    policy = ObservationLocalKeypointQualityPolicy(
        confidence_threshold=value.get("confidence_threshold"),
        minimum_valid_keypoints=value.get("minimum_valid_keypoints"),
        policy_version=value.get("schema_version"),
    )
    if dict(value) != policy.as_manifest():
        raise ValueError("Keypoint-quality policy differs from its frozen builder.")
    return policy


def _dimensions_from_logical(
    logical: Mapping[str, Any],
) -> KeypointQualityDimensions:
    raw = logical.get("dimensions")
    expected = {
        "n_frames",
        "n_frame_boundaries",
        "n_instances",
        "n_keypoints",
        "n_keypoint_metrics",
        "n_pose_metrics",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValueError("Keypoint-quality dimensions are not exact.")
    dimensions = KeypointQualityDimensions(
        n_frames=raw.get("n_frames"),
        n_instances=raw.get("n_instances"),
        n_keypoints=raw.get("n_keypoints"),
        n_keypoint_metrics=raw.get("n_keypoint_metrics"),
        n_pose_metrics=raw.get("n_pose_metrics"),
    )
    if dict(raw) != dimensions.as_manifest():
        raise ValueError("Keypoint-quality dimensions are not canonical.")
    return dimensions


def keypoint_quality_logical_content_document(
    arrays: Mapping[str, Any],
    *,
    dimensions: KeypointQualityDimensions,
    profile: KeypointQualityProfile,
    source: KeypointQualitySourceReference,
    source_arrays: Mapping[str, Any],
) -> dict[str, object]:
    KEYPOINT_QUALITY_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        profile=profile,
        source_keypoint_arrays=source_arrays,
    )
    declarations: dict[str, object] = {}
    for path in KEYPOINT_QUALITY_SCHEMA_V1.binding_paths:
        value = _array_values(arrays[path])
        declarations[path] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "digest_algorithm": KEYPOINT_QUALITY_ARRAY_DIGEST_ALGORITHM,
            "sha256": sha256_array(value),
        }
    return {
        "schema_id": KEYPOINT_QUALITY_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": KEYPOINT_QUALITY_LOGICAL_CONTENT_SCHEMA_VERSION,
        "logical_schema": {
            "id": KEYPOINT_QUALITY_SCHEMA_V1.schema_id,
            "version": KEYPOINT_QUALITY_SCHEMA_V1.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "source_manifest_digest": source.manifest_digest,
        "source_row_signatures_digest": source.keypoint_row_signatures_digest,
        "profile_digest": profile.profile_digest,
        "arrays": declarations,
    }


def keypoint_quality_metadata_declarations_digest(
    direct_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_by_path: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = {"", *KEYPOINT_QUALITY_SCHEMA_V1.binding_paths}
    if set(direct_by_path) != expected or set(consolidated_by_path) != expected:
        raise ValueError("Keypoint-quality metadata declaration paths are incomplete.")
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
                redacted.pop(KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE, None)
                declaration["attributes"] = redacted
        if direct != consolidated:
            raise ValueError(
                f"Direct and consolidated metadata differ at {path or '<run>'!r}."
            )
        normalized[path] = direct
    return canonical_json_sha256(
        {
            "scope": KEYPOINT_QUALITY_METADATA_DIGEST_SCOPE,
            "declarations": normalized,
        }
    )


def build_keypoint_quality_run_manifest(
    *,
    run_id: str,
    dimensions: KeypointQualityDimensions,
    profile: KeypointQualityProfile,
    policy: ObservationLocalKeypointQualityPolicy,
    source: KeypointQualitySourceReference,
    source_manifest: Mapping[str, Any],
    storage_plan: KeypointQualityStoragePlanSet,
    arrays: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Build the exact selector-ineligible persisted manifest envelope."""

    resolved_run_id = _require_run_id(run_id)
    if canonical_json_sha256(source_manifest) != source.manifest_digest:
        raise ValueError("Source manifest document differs from the bound digest.")
    if profile.policy_digest != policy.policy_digest:
        raise ValueError("Quality profile and producer policy digests differ.")
    if storage_plan.dimensions != dimensions:
        raise ValueError("Quality storage-plan dimensions differ.")
    content = keypoint_quality_logical_content_document(
        arrays,
        dimensions=dimensions,
        profile=profile,
        source=source,
        source_arrays=source_arrays,
    )
    metadata_digest = keypoint_quality_metadata_declarations_digest(
        direct_metadata_declarations,
        consolidated_by_path=consolidated_metadata_declarations,
    )
    payload: dict[str, object] = {
        "run_id": resolved_run_id,
        "stage": "keypoint_quality",
        "publication": {
            "artifact_class": "observation_local_quality_diagnostics",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": (
                KEYPOINT_QUALITY_METADATA_DIGEST_SCOPE
            ),
            "metadata_declarations_digest_algorithm": (
                CANONICAL_JSON_DIGEST_ALGORITHM
            ),
            "metadata_declarations_digest": metadata_digest,
        },
        "logical_schema": KEYPOINT_QUALITY_SCHEMA_V1.as_manifest(
            dimensions=dimensions,
            profile=profile,
            source=source,
        ),
        "storage_plan": storage_plan.as_manifest(),
        "source_keypoint_snapshot": source.as_manifest(),
        "policy": policy.as_manifest(),
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(content),
            "document": content,
        },
    }
    envelope = {
        "schema_id": KEYPOINT_QUALITY_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": KEYPOINT_QUALITY_RUN_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE,
        "persisted_path": KEYPOINT_QUALITY_RUN_MANIFEST_PERSISTED_PATH,
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
    KeypointQualityDimensions | None,
    KeypointQualityProfile | None,
    KeypointQualitySourceReference | None,
    ObservationLocalKeypointQualityPolicy | None,
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
        errors.append("quality run manifest envelope has unexpected fields")
    if (
        manifest.get("schema_id") != KEYPOINT_QUALITY_RUN_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version")
        != KEYPOINT_QUALITY_RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("persisted_attribute")
        != KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE
        or manifest.get("persisted_path")
        != KEYPOINT_QUALITY_RUN_MANIFEST_PERSISTED_PATH
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("quality run manifest envelope identity mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (
            [*errors, "quality run manifest payload must be an object"],
            None,
            None,
            None,
            None,
            None,
        )
    try:
        expected_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"quality run manifest is not strict JSON: {exc}")
    else:
        if manifest.get("payload_digest") != expected_digest:
            errors.append("quality run manifest payload_digest mismatch")
    expected_payload = {
        "run_id",
        "stage",
        "publication",
        "logical_schema",
        "storage_plan",
        "source_keypoint_snapshot",
        "policy",
        "logical_content",
    }
    if set(payload) != expected_payload:
        errors.append("quality run manifest payload has unexpected fields")
    try:
        _require_run_id(payload.get("run_id"))
    except ValueError as exc:
        errors.append(str(exc))
    if payload.get("stage") != "keypoint_quality":
        errors.append("quality run manifest stage mismatch")

    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("quality publication must be an object")
    else:
        expected_publication = {
            "artifact_class": "observation_local_quality_diagnostics",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": (
                KEYPOINT_QUALITY_METADATA_DIGEST_SCOPE
            ),
            "metadata_declarations_digest_algorithm": (
                CANONICAL_JSON_DIGEST_ALGORITHM
            ),
            "metadata_declarations_digest": publication.get(
                "metadata_declarations_digest"
            ),
        }
        if dict(publication) != expected_publication:
            errors.append("quality publication is not in exact persisted form")
        try:
            _require_sha256(
                publication.get("metadata_declarations_digest"),
                name="metadata_declarations_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))

    dimensions = None
    profile = None
    source = None
    logical = payload.get("logical_schema")
    if not isinstance(logical, Mapping):
        errors.append("quality logical_schema must be an object")
    else:
        try:
            dimensions = _dimensions_from_logical(logical)
            profile_value = logical.get("profile")
            source_value = logical.get("source")
            if not isinstance(profile_value, Mapping) or not isinstance(
                source_value, Mapping
            ):
                raise TypeError("Quality logical profile/source must be objects.")
            profile = quality_profile_from_manifest(profile_value)
            source = quality_source_from_manifest(source_value)
            expected_logical = KEYPOINT_QUALITY_SCHEMA_V1.as_manifest(
                dimensions=dimensions,
                profile=profile,
                source=source,
            )
            if dict(logical) != expected_logical:
                errors.append("quality logical_schema differs from frozen builder")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    source_value = payload.get("source_keypoint_snapshot")
    if isinstance(source_value, Mapping):
        try:
            top_source = quality_source_from_manifest(source_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if source is not None and top_source != source:
                errors.append("quality source declarations disagree")
            source = top_source
    else:
        errors.append("source_keypoint_snapshot must be an object")

    policy = None
    policy_value = payload.get("policy")
    if isinstance(policy_value, Mapping):
        try:
            policy = quality_policy_from_manifest(policy_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    else:
        errors.append("quality policy must be an object")
    if policy is not None and profile is not None:
        if policy.policy_digest != profile.policy_digest:
            errors.append("quality policy and profile digests differ")
        if profile.as_manifest() != quality_profile_for_policy(policy).as_manifest():
            errors.append("quality profile differs from the frozen producer policy")
    return errors, payload, dimensions, profile, source, policy


def validate_keypoint_quality_run_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Deeply validate the complete persisted envelope without array reads."""

    errors, payload, dimensions, profile, source, policy = (
        _parse_manifest_components(manifest)
    )
    if payload is None:
        return tuple(errors)

    storage = payload.get("storage_plan")
    if not isinstance(storage, Mapping):
        errors.append("quality storage_plan must be an object")
    elif dimensions is not None:
        raw_profile = storage.get("storage_profile")
        if not isinstance(raw_profile, Mapping):
            errors.append("quality storage profile must be an object")
        else:
            try:
                storage_profile = storage_profile_from_manifest(raw_profile)
                expected_storage = plan_keypoint_quality_storage(
                    dimensions, profile=storage_profile
                ).as_manifest()
            except (TypeError, ValueError) as exc:
                errors.append(f"cannot reconstruct quality storage plan: {exc}")
            else:
                if dict(storage) != expected_storage:
                    errors.append("quality storage plan differs from planner output")

    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping) or set(logical_content) != {
        "digest_algorithm",
        "digest",
        "document",
    }:
        errors.append("quality logical_content envelope is invalid")
    else:
        document = logical_content.get("document")
        if logical_content.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
            errors.append("quality logical_content digest algorithm mismatch")
        if not isinstance(document, Mapping):
            errors.append("quality logical_content document must be an object")
        else:
            try:
                digest = canonical_json_sha256(document)
            except (TypeError, ValueError) as exc:
                errors.append(f"quality logical_content is not strict JSON: {exc}")
            else:
                if logical_content.get("digest") != digest:
                    errors.append("quality logical_content digest mismatch")
            expected_document_fields = {
                "schema_id",
                "schema_version",
                "logical_schema",
                "dimensions",
                "source_manifest_digest",
                "source_row_signatures_digest",
                "profile_digest",
                "arrays",
            }
            if set(document) != expected_document_fields:
                errors.append("quality logical_content has unexpected fields")
            if (
                document.get("schema_id")
                != KEYPOINT_QUALITY_LOGICAL_CONTENT_SCHEMA_ID
                or document.get("schema_version")
                != KEYPOINT_QUALITY_LOGICAL_CONTENT_SCHEMA_VERSION
                or document.get("logical_schema")
                != {
                    "id": KEYPOINT_QUALITY_SCHEMA_V1.schema_id,
                    "version": KEYPOINT_QUALITY_SCHEMA_V1.schema_version,
                }
            ):
                errors.append("quality logical_content identity mismatch")
            if dimensions is not None and document.get(
                "dimensions"
            ) != dimensions.as_manifest():
                errors.append("quality logical_content dimensions mismatch")
            if source is not None:
                if document.get("source_manifest_digest") != source.manifest_digest:
                    errors.append("quality logical_content source digest mismatch")
                if document.get("source_row_signatures_digest") != (
                    source.keypoint_row_signatures_digest
                ):
                    errors.append("quality logical_content row digest mismatch")
            if profile is not None and document.get(
                "profile_digest"
            ) != profile.profile_digest:
                errors.append("quality logical_content profile digest mismatch")
            array_docs = document.get("arrays")
            if not isinstance(array_docs, Mapping) or set(array_docs) != set(
                KEYPOINT_QUALITY_SCHEMA_V1.binding_paths
            ):
                errors.append("quality logical_content array declarations mismatch")
            else:
                bindings = {
                    binding.path: binding
                    for binding in KEYPOINT_QUALITY_SCHEMA_V1.bindings
                }
                for path, item in array_docs.items():
                    if not isinstance(item, Mapping) or set(item) != {
                        "shape",
                        "dtype",
                        "digest_algorithm",
                        "sha256",
                    }:
                        errors.append(
                            f"quality logical_content declaration invalid at {path!r}"
                        )
                        continue
                    if dimensions is not None:
                        binding = bindings[path]
                        contract = KEYPOINT_QUALITY_SCHEMA_V1.contracts.resolve(
                            binding.contract_id,
                            binding.contract_version,
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
                            errors.append(
                                f"quality logical_content shape mismatch at {path}"
                            )
                        if item.get("dtype") != str(contract.dtype.numpy_dtype):
                            errors.append(
                                f"quality logical_content dtype mismatch at {path}"
                            )
                    if item.get("digest_algorithm") != (
                        KEYPOINT_QUALITY_ARRAY_DIGEST_ALGORITHM
                    ):
                        errors.append(f"quality array digest algorithm mismatch at {path}")
                    try:
                        _require_sha256(item.get("sha256"), name=f"{path} sha256")
                    except ValueError as exc:
                        errors.append(str(exc))

    if policy is None:
        errors.append("quality policy could not be reconstructed")
    return tuple(errors)


def validate_keypoint_quality_publication(
    manifest: Mapping[str, Any],
    *,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recompute decoded, source, metadata, and physical publication evidence."""

    errors = list(validate_keypoint_quality_run_manifest(manifest))
    _, payload, dimensions, profile, source, _ = _parse_manifest_components(manifest)
    if (
        payload is None
        or dimensions is None
        or profile is None
        or source is None
    ):
        return (*errors, "quality publication manifest components are invalid")
    try:
        source_manifest_digest = canonical_json_sha256(source_manifest)
    except (TypeError, ValueError) as exc:
        errors.append(f"quality source manifest is not strict JSON: {exc}")
    else:
        if source_manifest_digest != source.manifest_digest:
            errors.append("quality source manifest digest mismatch")
    try:
        observed_signature_digest = sha256_array(
            _array_values(source_arrays["keypoint_row_signature"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"quality source row signatures are invalid: {exc}")
    else:
        if observed_signature_digest != source.keypoint_row_signatures_digest:
            errors.append("quality source row-signature digest mismatch")

    try:
        content = keypoint_quality_logical_content_document(
            arrays,
            dimensions=dimensions,
            profile=profile,
            source=source,
            source_arrays=source_arrays,
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"quality logical array validation failed: {exc}")
    else:
        if content != payload["logical_content"]["document"]:
            errors.append("quality logical_content differs from decoded arrays")

    try:
        metadata_digest = keypoint_quality_metadata_declarations_digest(
            direct_metadata_declarations,
            consolidated_by_path=consolidated_metadata_declarations,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"quality metadata declaration validation failed: {exc}")
    else:
        if metadata_digest != payload["publication"].get(
            "metadata_declarations_digest"
        ):
            errors.append("quality metadata declaration digest mismatch")

    storage = payload.get("storage_plan")
    raw_profile = storage.get("storage_profile") if isinstance(storage, Mapping) else None
    try:
        if not isinstance(raw_profile, Mapping):
            raise ValueError("quality storage profile is missing")
        storage_profile = storage_profile_from_manifest(raw_profile)
        plans = plan_keypoint_quality_storage(dimensions, profile=storage_profile)
    except (TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct quality physical plan: {exc}")
    else:
        bindings = {
            binding.path: binding for binding in KEYPOINT_QUALITY_SCHEMA_V1.bindings
        }
        for entry in plans.entries:
            declaration = direct_metadata_declarations.get(entry.rule.path)
            if not isinstance(declaration, Mapping):
                errors.append(f"missing direct metadata at {entry.rule.path}")
                continue
            binding = bindings[entry.rule.path]
            contract = KEYPOINT_QUALITY_SCHEMA_V1.contracts.resolve(
                binding.contract_id, binding.contract_version
            )
            errors.extend(
                f"quality physical metadata at {entry.rule.path}: {error}"
                for error in validate_array_metadata_declaration_from_plan(
                    declaration,
                    contract=contract,
                    plan=entry.plan,
                    fill_value=0,
                )
            )
    return tuple(errors)


__all__ = [
    "KEYPOINT_QUALITY_ARRAY_DIGEST_ALGORITHM",
    "KEYPOINT_QUALITY_LOGICAL_CONTENT_SCHEMA_ID",
    "KEYPOINT_QUALITY_LOGICAL_CONTENT_SCHEMA_VERSION",
    "KEYPOINT_QUALITY_METADATA_DIGEST_SCOPE",
    "KEYPOINT_QUALITY_RUN_MANIFEST_ATTRIBUTE",
    "KEYPOINT_QUALITY_RUN_MANIFEST_PERSISTED_PATH",
    "KEYPOINT_QUALITY_RUN_MANIFEST_SCHEMA_ID",
    "KEYPOINT_QUALITY_RUN_MANIFEST_SCHEMA_VERSION",
    "build_keypoint_quality_run_manifest",
    "keypoint_quality_logical_content_document",
    "keypoint_quality_metadata_declarations_digest",
    "quality_policy_from_manifest",
    "quality_profile_from_manifest",
    "quality_source_from_manifest",
    "validate_keypoint_quality_publication",
    "validate_keypoint_quality_run_manifest",
]
