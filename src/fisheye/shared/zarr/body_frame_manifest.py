"""Strict manifest and publication gate for body-frame v1."""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.zarr.array_factory import (
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.body_frame_producer import (
    BODY_FRAME_RECIPE_SCHEMA_ID,
    BODY_FRAME_RECIPE_SCHEMA_VERSION,
    KEYPOINT_HEAD_AXIS_RECIPE_ID,
    KEYPOINT_HEADING_COMPUTATION_SOURCE,
    BodyFrameSourceReference,
    KeypointBodyFrameRecipe,
    derive_body_frame_geometry,
)
from fisheye.shared.zarr.body_frame_schema import (
    BODY_FRAME_SCHEMA_V1,
    BodyFrameDimensions,
)
from fisheye.shared.zarr.body_frame_storage import (
    BodyFrameStoragePlanSet,
    plan_body_frame_storage,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest


BODY_FRAME_RUN_MANIFEST_SCHEMA_ID = "palette.body_frame.run_manifest"
BODY_FRAME_RUN_MANIFEST_SCHEMA_VERSION = 1
BODY_FRAME_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
BODY_FRAME_RUN_MANIFEST_PERSISTED_PATH = (
    "analysis/body_frame_runs/<run>/zarr.json.attributes.run_manifest"
)
BODY_FRAME_LOGICAL_CONTENT_SCHEMA_ID = "palette.body_frame.logical_content"
BODY_FRAME_LOGICAL_CONTENT_SCHEMA_VERSION = 1
BODY_FRAME_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_bytes_v1"
BODY_FRAME_METADATA_DIGEST_SCOPE = (
    "exact_group_and_array_declarations_with_attributes_redacting_only_run_manifest"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


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


def body_frame_source_from_manifest(
    value: Mapping[str, Any],
) -> BodyFrameSourceReference:
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
        raise ValueError("Body-frame source has an unexpected field set.")
    source = BodyFrameSourceReference(
        stage=value.get("stage"),
        run_name=value.get("run_name"),
        manifest_digest=value.get("manifest_digest"),
        skeleton_id=value.get("skeleton_id"),
        skeleton_digest=value.get("skeleton_digest"),
        keypoint_row_signatures_digest=value.get("keypoint_row_signatures_digest"),
    )
    if dict(value) != source.as_manifest():
        raise ValueError("Body-frame source differs from its frozen builder.")
    return source


def body_frame_recipe_from_manifest(
    value: Mapping[str, Any],
) -> KeypointBodyFrameRecipe:
    expected = {
        "schema_id",
        "schema_version",
        "recipe_id",
        "skeleton_digest",
        "heading_computation_source",
        "heading_computation",
        "heading_computation_digest",
        "keypoint_indices",
        "coordinate_source",
        "origin",
        "forward_axis",
        "left_axis",
        "axis_handedness",
        "heading_deg",
        "invalid_geometry",
        "recipe_digest",
    }
    if set(value) != expected:
        raise ValueError("Body-frame recipe has an unexpected field set.")
    indices = value.get("keypoint_indices")
    if not isinstance(indices, Mapping) or set(indices) != {
        "swim_bladder",
        "eye_left",
        "eye_right",
    }:
        raise ValueError("Body-frame recipe keypoint indices are not exact.")
    if (
        value.get("schema_id") != BODY_FRAME_RECIPE_SCHEMA_ID
        or value.get("schema_version") != BODY_FRAME_RECIPE_SCHEMA_VERSION
        or value.get("recipe_id") != KEYPOINT_HEAD_AXIS_RECIPE_ID
        or value.get("heading_computation_source")
        != KEYPOINT_HEADING_COMPUTATION_SOURCE
    ):
        raise ValueError("Body-frame recipe identity mismatch.")
    recipe = KeypointBodyFrameRecipe(
        swim_bladder_index=indices.get("swim_bladder"),
        eye_left_index=indices.get("eye_left"),
        eye_right_index=indices.get("eye_right"),
        skeleton_digest=value.get("skeleton_digest"),
        heading_computation=value.get("heading_computation"),
        heading_computation_source=value.get("heading_computation_source"),
        recipe_version=value.get("schema_version"),
    )
    if dict(value) != recipe.as_manifest():
        raise ValueError("Body-frame recipe differs from its frozen builder.")
    return recipe


def _dimensions_from_manifest(value: object) -> BodyFrameDimensions:
    if not isinstance(value, Mapping) or set(value) != {
        "n_frames",
        "n_frame_boundaries",
        "n_instances",
    }:
        raise ValueError("Body-frame dimensions are not exact.")
    dimensions = BodyFrameDimensions(
        n_frames=value.get("n_frames"),
        n_instances=value.get("n_instances"),
    )
    if dict(value) != dimensions.as_manifest():
        raise ValueError("Body-frame dimensions are not canonical.")
    return dimensions


def body_frame_logical_content_document(
    arrays: Mapping[str, Any],
    *,
    dimensions: BodyFrameDimensions,
    source: BodyFrameSourceReference,
    recipe: KeypointBodyFrameRecipe,
    source_arrays: Mapping[str, Any],
) -> dict[str, object]:
    BODY_FRAME_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        source_keypoint_arrays=source_arrays,
    )
    declarations: dict[str, object] = {}
    for path in BODY_FRAME_SCHEMA_V1.binding_paths:
        value = _array_values(arrays[path])
        declarations[path] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "digest_algorithm": BODY_FRAME_ARRAY_DIGEST_ALGORITHM,
            "sha256": sha256_array(value),
        }
    return {
        "schema_id": BODY_FRAME_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": BODY_FRAME_LOGICAL_CONTENT_SCHEMA_VERSION,
        "logical_schema": {
            "id": BODY_FRAME_SCHEMA_V1.schema_id,
            "version": BODY_FRAME_SCHEMA_V1.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "source_manifest_digest": source.manifest_digest,
        "source_row_signatures_digest": source.keypoint_row_signatures_digest,
        "recipe_digest": recipe.recipe_digest,
        "arrays": declarations,
    }


def body_frame_metadata_declarations_digest(
    direct_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_by_path: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = {"", *BODY_FRAME_SCHEMA_V1.binding_paths}
    if set(direct_by_path) != expected or set(consolidated_by_path) != expected:
        raise ValueError("Body-frame metadata declaration paths are incomplete.")
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
                    raise ValueError("Body-frame run attributes must be an object.")
                redacted = dict(attributes)
                redacted.pop(BODY_FRAME_RUN_MANIFEST_ATTRIBUTE, None)
                declaration["attributes"] = redacted
        if direct != consolidated:
            raise ValueError(
                f"Direct and consolidated metadata differ at {path or '<run>'!r}."
            )
        normalized[path] = direct
    return canonical_json_sha256(
        {"scope": BODY_FRAME_METADATA_DIGEST_SCOPE, "declarations": normalized}
    )


def build_body_frame_run_manifest(
    *,
    run_id: str,
    dimensions: BodyFrameDimensions,
    source: BodyFrameSourceReference,
    source_manifest: Mapping[str, Any],
    recipe: KeypointBodyFrameRecipe,
    storage_plan: BodyFrameStoragePlanSet,
    arrays: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    resolved_run_id = _require_run_id(run_id)
    if canonical_json_sha256(source_manifest) != source.manifest_digest:
        raise ValueError("Source manifest differs from the body-frame binding.")
    if source.skeleton_digest != recipe.skeleton_digest:
        raise ValueError("Body-frame source and recipe skeleton digests differ.")
    if storage_plan.dimensions != dimensions:
        raise ValueError("Body-frame storage-plan dimensions differ.")
    content = body_frame_logical_content_document(
        arrays,
        dimensions=dimensions,
        source=source,
        recipe=recipe,
        source_arrays=source_arrays,
    )
    metadata_digest = body_frame_metadata_declarations_digest(
        direct_metadata_declarations,
        consolidated_by_path=consolidated_metadata_declarations,
    )
    payload: dict[str, object] = {
        "run_id": resolved_run_id,
        "stage": "body_frame",
        "publication": {
            "artifact_class": "derived_keypoint_body_frame_cache",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "keypoint_authority": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": BODY_FRAME_METADATA_DIGEST_SCOPE,
            "metadata_declarations_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "metadata_declarations_digest": metadata_digest,
        },
        "logical_schema": BODY_FRAME_SCHEMA_V1.as_manifest(dimensions=dimensions),
        "storage_plan": storage_plan.as_manifest(),
        "source_keypoint_snapshot": source.as_manifest(),
        "heading_recipe": recipe.as_manifest(),
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(content),
            "document": content,
        },
    }
    envelope = {
        "schema_id": BODY_FRAME_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": BODY_FRAME_RUN_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": BODY_FRAME_RUN_MANIFEST_ATTRIBUTE,
        "persisted_path": BODY_FRAME_RUN_MANIFEST_PERSISTED_PATH,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def _parse_components(manifest: Mapping[str, Any]):  # type: ignore[no-untyped-def]
    errors: list[str] = []
    if set(manifest) != {
        "schema_id",
        "schema_version",
        "persisted_attribute",
        "persisted_path",
        "digest_algorithm",
        "payload_digest",
        "payload",
    }:
        errors.append("body-frame manifest envelope has unexpected fields")
    if (
        manifest.get("schema_id") != BODY_FRAME_RUN_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != BODY_FRAME_RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("persisted_attribute") != BODY_FRAME_RUN_MANIFEST_ATTRIBUTE
        or manifest.get("persisted_path") != BODY_FRAME_RUN_MANIFEST_PERSISTED_PATH
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("body-frame manifest envelope identity mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (
            [*errors, "body-frame manifest payload must be an object"],
            None,
            None,
            None,
            None,
        )
    try:
        digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        errors.append(f"body-frame manifest is not strict JSON: {exc}")
    else:
        if manifest.get("payload_digest") != digest:
            errors.append("body-frame manifest payload_digest mismatch")
    if set(payload) != {
        "run_id",
        "stage",
        "publication",
        "logical_schema",
        "storage_plan",
        "source_keypoint_snapshot",
        "heading_recipe",
        "logical_content",
    }:
        errors.append("body-frame manifest payload has unexpected fields")
    try:
        _require_run_id(payload.get("run_id"))
    except ValueError as exc:
        errors.append(str(exc))
    if payload.get("stage") != "body_frame":
        errors.append("body-frame manifest stage mismatch")

    source = recipe = dimensions = None
    source_value = payload.get("source_keypoint_snapshot")
    recipe_value = payload.get("heading_recipe")
    logical = payload.get("logical_schema")
    try:
        if not isinstance(source_value, Mapping):
            raise TypeError("Body-frame source must be an object.")
        source = body_frame_source_from_manifest(source_value)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    try:
        if not isinstance(recipe_value, Mapping):
            raise TypeError("Body-frame recipe must be an object.")
        recipe = body_frame_recipe_from_manifest(recipe_value)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    if source is not None and recipe is not None:
        if source.skeleton_digest != recipe.skeleton_digest:
            errors.append("body-frame source and recipe skeleton digests differ")
    try:
        if not isinstance(logical, Mapping):
            raise TypeError("Body-frame logical schema must be an object.")
        dimensions = _dimensions_from_manifest(logical.get("dimensions"))
        if dict(logical) != BODY_FRAME_SCHEMA_V1.as_manifest(dimensions=dimensions):
            errors.append("body-frame logical schema differs from frozen builder")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors, payload, dimensions, source, recipe


def validate_body_frame_run_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    errors, payload, dimensions, source, recipe = _parse_components(manifest)
    if payload is None:
        return tuple(errors)
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("body-frame publication must be an object")
    else:
        expected = {
            "artifact_class": "derived_keypoint_body_frame_cache",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "keypoint_authority": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": BODY_FRAME_METADATA_DIGEST_SCOPE,
            "metadata_declarations_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "metadata_declarations_digest": publication.get(
                "metadata_declarations_digest"
            ),
        }
        if dict(publication) != expected:
            errors.append("body-frame publication is not in exact persisted form")
        try:
            _require_sha256(
                publication.get("metadata_declarations_digest"),
                name="metadata_declarations_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))

    storage = payload.get("storage_plan")
    if not isinstance(storage, Mapping):
        errors.append("body-frame storage plan must be an object")
    elif dimensions is not None:
        profile = storage.get("storage_profile")
        try:
            if not isinstance(profile, Mapping):
                raise TypeError("Body-frame storage profile must be an object.")
            expected_storage = plan_body_frame_storage(
                dimensions, profile=storage_profile_from_manifest(profile)
            ).as_manifest()
            if dict(storage) != expected_storage:
                errors.append("body-frame storage plan differs from planner output")
        except (TypeError, ValueError) as exc:
            errors.append(f"cannot reconstruct body-frame storage plan: {exc}")

    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping) or set(logical_content) != {
        "digest_algorithm",
        "digest",
        "document",
    }:
        errors.append("body-frame logical content envelope is invalid")
        return tuple(errors)
    document = logical_content.get("document")
    if logical_content.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("body-frame logical content digest algorithm mismatch")
    if not isinstance(document, Mapping):
        errors.append("body-frame logical content document must be an object")
        return tuple(errors)
    try:
        document_digest = canonical_json_sha256(document)
    except (TypeError, ValueError) as exc:
        errors.append(f"body-frame logical content is not strict JSON: {exc}")
    else:
        if logical_content.get("digest") != document_digest:
            errors.append("body-frame logical content digest mismatch")
    if set(document) != {
        "schema_id",
        "schema_version",
        "logical_schema",
        "dimensions",
        "source_manifest_digest",
        "source_row_signatures_digest",
        "recipe_digest",
        "arrays",
    }:
        errors.append("body-frame logical content has unexpected fields")
    if (
        document.get("schema_id") != BODY_FRAME_LOGICAL_CONTENT_SCHEMA_ID
        or document.get("schema_version") != BODY_FRAME_LOGICAL_CONTENT_SCHEMA_VERSION
        or document.get("logical_schema")
        != {
            "id": BODY_FRAME_SCHEMA_V1.schema_id,
            "version": BODY_FRAME_SCHEMA_V1.schema_version,
        }
    ):
        errors.append("body-frame logical content identity mismatch")
    if (
        dimensions is not None
        and document.get("dimensions") != dimensions.as_manifest()
    ):
        errors.append("body-frame logical content dimensions mismatch")
    if source is not None:
        if document.get("source_manifest_digest") != source.manifest_digest:
            errors.append("body-frame logical content source digest mismatch")
        if (
            document.get("source_row_signatures_digest")
            != source.keypoint_row_signatures_digest
        ):
            errors.append("body-frame logical content row-signature digest mismatch")
    if recipe is not None and document.get("recipe_digest") != recipe.recipe_digest:
        errors.append("body-frame logical content recipe digest mismatch")
    arrays = document.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(
        BODY_FRAME_SCHEMA_V1.binding_paths
    ):
        errors.append("body-frame logical content array declarations mismatch")
    elif dimensions is not None:
        bindings = {binding.path: binding for binding in BODY_FRAME_SCHEMA_V1.bindings}
        for path, item in arrays.items():
            if not isinstance(item, Mapping) or set(item) != {
                "shape",
                "dtype",
                "digest_algorithm",
                "sha256",
            }:
                errors.append(f"body-frame logical declaration invalid at {path}")
                continue
            contract = BODY_FRAME_SCHEMA_V1.contracts.resolve(
                bindings[path].contract_id, bindings[path].contract_version
            )
            expected_shape = [
                axis if isinstance(axis, int) else dimensions.contract_dimensions[axis]
                for axis in contract.shape_template
            ]
            if item.get("shape") != expected_shape or item.get("dtype") != str(
                contract.dtype.numpy_dtype
            ):
                errors.append(f"body-frame logical declaration mismatch at {path}")
            if item.get("digest_algorithm") != BODY_FRAME_ARRAY_DIGEST_ALGORITHM:
                errors.append(f"body-frame array digest algorithm mismatch at {path}")
            try:
                _require_sha256(item.get("sha256"), name=f"{path} sha256")
            except ValueError as exc:
                errors.append(str(exc))
    return tuple(errors)


def validate_body_frame_publication(
    manifest: Mapping[str, Any],
    *,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    errors = list(validate_body_frame_run_manifest(manifest))
    _, payload, dimensions, source, recipe = _parse_components(manifest)
    if payload is None or dimensions is None or source is None or recipe is None:
        return (*errors, "body-frame manifest components are invalid")
    try:
        source_manifest_digest = canonical_json_sha256(source_manifest)
    except (TypeError, ValueError) as exc:
        errors.append(f"body-frame source manifest is not strict JSON: {exc}")
    else:
        if source_manifest_digest != source.manifest_digest:
            errors.append("body-frame source manifest digest mismatch")
    try:
        if sha256_array(_array_values(source_arrays["keypoint_row_signature"])) != (
            source.keypoint_row_signatures_digest
        ):
            errors.append("body-frame source row-signature digest mismatch")
        content = body_frame_logical_content_document(
            arrays,
            dimensions=dimensions,
            source=source,
            recipe=recipe,
            source_arrays=source_arrays,
        )
        if content != payload["logical_content"]["document"]:
            errors.append("body-frame logical content differs from decoded arrays")
        derived = derive_body_frame_geometry(source_arrays, recipe=recipe)
        for path, expected in derived.items():
            if not np.array_equal(
                _array_values(arrays[path]), expected, equal_nan=True
            ):
                errors.append(f"body-frame derived geometry mismatch at {path}")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"body-frame logical validation failed: {exc}")

    try:
        digest = body_frame_metadata_declarations_digest(
            direct_metadata_declarations,
            consolidated_by_path=consolidated_metadata_declarations,
        )
        if digest != payload["publication"].get("metadata_declarations_digest"):
            errors.append("body-frame metadata declaration digest mismatch")
    except (TypeError, ValueError) as exc:
        errors.append(f"body-frame metadata validation failed: {exc}")

    storage = payload.get("storage_plan")
    profile = storage.get("storage_profile") if isinstance(storage, Mapping) else None
    try:
        if not isinstance(profile, Mapping):
            raise ValueError("body-frame storage profile is missing")
        plans = plan_body_frame_storage(
            dimensions, profile=storage_profile_from_manifest(profile)
        )
        bindings = {binding.path: binding for binding in BODY_FRAME_SCHEMA_V1.bindings}
        for entry in plans.entries:
            declaration = direct_metadata_declarations.get(entry.rule.path)
            if not isinstance(declaration, Mapping):
                errors.append(f"missing direct metadata at {entry.rule.path}")
                continue
            contract = BODY_FRAME_SCHEMA_V1.contracts.resolve(
                bindings[entry.rule.path].contract_id,
                bindings[entry.rule.path].contract_version,
            )
            errors.extend(
                f"body-frame physical metadata at {entry.rule.path}: {error}"
                for error in validate_array_metadata_declaration_from_plan(
                    declaration,
                    contract=contract,
                    plan=entry.plan,
                    fill_value=0,
                )
            )
    except (TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct body-frame physical plan: {exc}")
    return tuple(errors)


__all__ = [
    "BODY_FRAME_ARRAY_DIGEST_ALGORITHM",
    "BODY_FRAME_LOGICAL_CONTENT_SCHEMA_ID",
    "BODY_FRAME_LOGICAL_CONTENT_SCHEMA_VERSION",
    "BODY_FRAME_METADATA_DIGEST_SCOPE",
    "BODY_FRAME_RUN_MANIFEST_ATTRIBUTE",
    "BODY_FRAME_RUN_MANIFEST_PERSISTED_PATH",
    "BODY_FRAME_RUN_MANIFEST_SCHEMA_ID",
    "BODY_FRAME_RUN_MANIFEST_SCHEMA_VERSION",
    "body_frame_logical_content_document",
    "body_frame_metadata_declarations_digest",
    "body_frame_recipe_from_manifest",
    "body_frame_source_from_manifest",
    "build_body_frame_run_manifest",
    "validate_body_frame_publication",
    "validate_body_frame_run_manifest",
]
