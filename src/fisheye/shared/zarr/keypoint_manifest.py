"""Strict manifest and publication gate for raw keypoint-v2 snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping

import numpy as np

from fisheye.shared.pose_model_schema_binding import (
    validate_pose_model_schema_binding,
)
from fisheye.shared.zarr.array_factory import (
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.coordinate_manifest import (
    build_coordinate_catalog_envelope,
    validate_coordinate_catalog_envelope,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.keypoint_schema import KEYPOINT_SCHEMA_V2, KeypointDimensions
from fisheye.shared.zarr.keypoint_storage import (
    KeypointStoragePlanSet,
    plan_keypoint_storage,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest
from fisheye.shared.zarr.training_keypoint_crop_source import (
    TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID,
    validate_training_keypoint_crop_source_manifest,
)


KEYPOINT_RUN_MANIFEST_SCHEMA_ID = "palette.keypoint.run_manifest"
KEYPOINT_RUN_MANIFEST_SCHEMA_VERSION = 1
KEYPOINT_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
KEYPOINT_RUN_MANIFEST_PERSISTED_PATH = (
    "keypoints_runs/<run>/zarr.json.attributes.run_manifest"
)
KEYPOINT_LOGICAL_CONTENT_SCHEMA_ID = "palette.keypoint.logical_content"
KEYPOINT_LOGICAL_CONTENT_SCHEMA_VERSION = 1
KEYPOINT_CROP_SOURCE_SCHEMA_ID = "palette.keypoint.crop_source"
KEYPOINT_CROP_SOURCE_SCHEMA_VERSION = 1
KEYPOINT_PREPROCESSING_SCHEMA_ID = "palette.keypoint.preprocessing"
KEYPOINT_PREPROCESSING_SCHEMA_VERSION = 1
KEYPOINT_SKELETON_SEMANTICS_SCHEMA_ID = "palette.keypoint.skeleton_semantics"
KEYPOINT_SKELETON_SEMANTICS_SCHEMA_VERSION = 1
KEYPOINT_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_bytes_v1"
KEYPOINT_METADATA_DIGEST_SCOPE = (
    "exact_group_and_array_declarations_with_attributes_redacting_only_run_manifest"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256.")
    return normalized


def _require_identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be lowercase snake_case.")
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


@dataclass(frozen=True)
class KeypointCropSourceReference:
    """Exact coordinate-catalog crop-v2 snapshot used for keypoint rows."""

    run_id: str
    manifest_digest: str
    logical_content_digest: str
    row_signatures_digest: str
    coordinate_catalog_digest: str
    n_frames: int
    n_instances: int
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_run_id(self.run_id))
        for name in (
            "manifest_digest",
            "logical_content_digest",
            "row_signatures_digest",
            "coordinate_catalog_digest",
        ):
            object.__setattr__(
                self, name, _require_sha256(getattr(self, name), name=name)
            )
        for name in ("n_frames", "n_instances", "source_width", "source_height"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative exact integer.")
        if self.n_frames <= 0 or self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("Crop frame and source dimensions must be positive.")

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": KEYPOINT_CROP_SOURCE_SCHEMA_ID,
            "schema_version": KEYPOINT_CROP_SOURCE_SCHEMA_VERSION,
            "stage": "crop",
            "run_id": self.run_id,
            "run_path": f"crop_runs/{self.run_id}",
            "manifest_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "manifest_digest": self.manifest_digest,
            "logical_content_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "logical_content_digest": self.logical_content_digest,
            "row_signatures_digest_algorithm": KEYPOINT_ARRAY_DIGEST_ALGORITHM,
            "row_signatures_digest": self.row_signatures_digest,
            "coordinate_catalog_digest": self.coordinate_catalog_digest,
            "dimensions": {
                "n_frames": self.n_frames,
                "n_instances": self.n_instances,
                "source_width": self.source_width,
                "source_height": self.source_height,
            },
        }


def keypoint_crop_source_from_manifest(
    manifest: Mapping[str, Any],
) -> KeypointCropSourceReference:
    if manifest.get("schema_id") == TRAINING_KEYPOINT_CROP_SOURCE_SCHEMA_ID:
        errors = validate_training_keypoint_crop_source_manifest(manifest)
        if errors:
            raise ValueError(
                "Training crop source manifest is invalid: " + "; ".join(errors)
            )
        payload = manifest["payload"]
        dimensions = payload["dimensions"]
        return KeypointCropSourceReference(
            run_id=payload["run_id"],
            manifest_digest=canonical_json_sha256(manifest),
            logical_content_digest=payload["logical_content"]["digest"],
            row_signatures_digest=payload["row_signatures_digest"],
            coordinate_catalog_digest=payload["coordinate_contract"]["digest"],
            n_frames=dimensions["n_frames"],
            n_instances=dimensions["n_instances"],
            source_width=dimensions["source_width"],
            source_height=dimensions["source_height"],
        )
    errors = validate_crop_run_manifest(manifest)
    if errors:
        raise ValueError("Crop manifest is invalid: " + "; ".join(errors))
    if manifest.get("schema_version") != CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Raw keypoint v2 requires a coordinate-catalog crop-v2 run.")
    payload = manifest["payload"]
    logical = payload["logical_content"]
    declarations = logical["document"]["arrays"]
    dimensions = payload["logical_schema"]["dimensions"]
    return KeypointCropSourceReference(
        run_id=payload["run_id"],
        manifest_digest=canonical_json_sha256(manifest),
        logical_content_digest=logical["digest"],
        row_signatures_digest=declarations["source_row_signature"]["sha256"],
        coordinate_catalog_digest=payload["coordinate_contract"]["digest"],
        n_frames=dimensions["n_frames"],
        n_instances=dimensions["n_instances"],
        source_width=dimensions["source_width"],
        source_height=dimensions["source_height"],
    )


def keypoint_crop_source_from_persisted(
    value: Mapping[str, Any],
) -> KeypointCropSourceReference:
    expected = set(
        KeypointCropSourceReference(
            run_id="placeholder",
            manifest_digest="0" * 64,
            logical_content_digest="0" * 64,
            row_signatures_digest="0" * 64,
            coordinate_catalog_digest="0" * 64,
            n_frames=1,
            n_instances=0,
            source_width=1,
            source_height=1,
        ).as_manifest()
    )
    if set(value) != expected:
        raise ValueError("Keypoint crop source has an unexpected field set.")
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != {
        "n_frames",
        "n_instances",
        "source_width",
        "source_height",
    }:
        raise ValueError("Keypoint crop source dimensions are invalid.")
    reference = KeypointCropSourceReference(
        run_id=value.get("run_id"),
        manifest_digest=value.get("manifest_digest"),
        logical_content_digest=value.get("logical_content_digest"),
        row_signatures_digest=value.get("row_signatures_digest"),
        coordinate_catalog_digest=value.get("coordinate_catalog_digest"),
        n_frames=dimensions.get("n_frames"),
        n_instances=dimensions.get("n_instances"),
        source_width=dimensions.get("source_width"),
        source_height=dimensions.get("source_height"),
    )
    if dict(value) != reference.as_manifest():
        raise ValueError("Keypoint crop source differs from its frozen builder.")
    return reference


@dataclass(frozen=True)
class KeypointPreprocessingReference:
    profile_id: str
    profile_version: int
    input_mode: str
    document: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_id", _require_identifier(self.profile_id, name="profile_id")
        )
        object.__setattr__(
            self, "input_mode", _require_identifier(self.input_mode, name="input_mode")
        )
        if type(self.profile_version) is not int or self.profile_version <= 0:
            raise ValueError("profile_version must be a positive exact integer.")
        normalized = json.loads(
            canonical_json_bytes(dict(self.document)).decode("utf-8")
        )
        object.__setattr__(self, "document", normalized)

    def payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "input_mode": self.input_mode,
            "document": dict(self.document),
        }

    @property
    def preprocessing_digest(self) -> str:
        return canonical_json_sha256(self.payload())

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": KEYPOINT_PREPROCESSING_SCHEMA_ID,
            "schema_version": KEYPOINT_PREPROCESSING_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "input_mode": self.input_mode,
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "preprocessing_digest": self.preprocessing_digest,
            "document": dict(self.document),
        }


def keypoint_preprocessing_from_manifest(
    value: Mapping[str, Any],
) -> KeypointPreprocessingReference:
    if set(value) != {
        "schema_id",
        "schema_version",
        "profile_id",
        "profile_version",
        "input_mode",
        "digest_algorithm",
        "preprocessing_digest",
        "document",
    }:
        raise ValueError("Keypoint preprocessing has an unexpected field set.")
    if (
        value.get("schema_id") != KEYPOINT_PREPROCESSING_SCHEMA_ID
        or value.get("schema_version") != KEYPOINT_PREPROCESSING_SCHEMA_VERSION
        or value.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
        or not isinstance(value.get("document"), Mapping)
    ):
        raise ValueError("Keypoint preprocessing identity mismatch.")
    reference = KeypointPreprocessingReference(
        profile_id=value.get("profile_id"),
        profile_version=value.get("profile_version"),
        input_mode=value.get("input_mode"),
        document=value.get("document"),
    )
    if dict(value) != reference.as_manifest():
        raise ValueError("Keypoint preprocessing differs from its frozen builder.")
    return reference


def _validated_pose_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    model = value.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Pose-model binding lacks model identity.")
    return validate_pose_model_schema_binding(
        dict(value), expected_model_sha256=model.get("sha256")
    )


def keypoint_skeleton_digest(binding: Mapping[str, Any]) -> str:
    return canonical_json_sha256(keypoint_skeleton_document(binding))


def keypoint_skeleton_document(binding: Mapping[str, Any]) -> dict[str, object]:
    """Return model/source-independent ordered skeleton semantics."""

    validated = _validated_pose_binding(binding)
    schema = validated["pose_schema"]
    metadata = schema["metadata"]
    return {
        "schema_id": KEYPOINT_SKELETON_SEMANTICS_SCHEMA_ID,
        "schema_version": KEYPOINT_SKELETON_SEMANTICS_SCHEMA_VERSION,
        "skeleton_id": schema["skeleton_id"],
        "kpt_shape": list(schema["kpt_shape"]),
        "keypoint_labels": list(schema["keypoint_labels"]),
        "nodes": list(schema["nodes"]),
        "edges": list(schema["edges"]),
        "heading_computation": metadata["heading_computation"],
        "heading_computation_source": metadata["heading_computation_source"],
    }


def _dimensions_from_manifest(value: object) -> KeypointDimensions:
    if not isinstance(value, Mapping) or set(value) != {
        "n_frames",
        "n_frame_boundaries",
        "n_instances",
        "n_keypoints",
        "source_width",
        "source_height",
    }:
        raise ValueError("Keypoint dimensions are not exact.")
    dimensions = KeypointDimensions(
        n_frames=value.get("n_frames"),
        n_instances=value.get("n_instances"),
        n_keypoints=value.get("n_keypoints"),
        source_width=value.get("source_width"),
        source_height=value.get("source_height"),
    )
    if dict(value) != dimensions.as_manifest():
        raise ValueError("Keypoint dimensions are not canonical.")
    return dimensions


def keypoint_logical_content_document(
    arrays: Mapping[str, Any],
    *,
    dimensions: KeypointDimensions,
    crop_source: KeypointCropSourceReference,
    source_crop_arrays: Mapping[str, Any],
    pose_binding: Mapping[str, Any],
    preprocessing: KeypointPreprocessingReference,
) -> dict[str, object]:
    binding = _validated_pose_binding(pose_binding)
    skeleton_digest = keypoint_skeleton_digest(binding)
    if binding["pose_schema"]["kpt_shape"][0] != dimensions.n_keypoints:
        raise ValueError(
            "Pose-model binding cardinality differs from keypoint dimensions."
        )
    if (
        crop_source.n_frames != dimensions.n_frames
        or crop_source.n_instances != dimensions.n_instances
        or crop_source.source_width != dimensions.source_width
        or crop_source.source_height != dimensions.source_height
    ):
        raise ValueError("Crop and keypoint dimensions differ.")
    KEYPOINT_SCHEMA_V2.require(
        arrays,
        dimensions=dimensions,
        source_crop_arrays=source_crop_arrays,
        skeleton_digest=skeleton_digest,
    )
    declarations: dict[str, object] = {}
    for path in KEYPOINT_SCHEMA_V2.binding_paths:
        values = _array_values(arrays[path])
        declarations[path] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "digest_algorithm": KEYPOINT_ARRAY_DIGEST_ALGORITHM,
            "sha256": sha256_array(values),
        }
    return {
        "schema_id": KEYPOINT_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": KEYPOINT_LOGICAL_CONTENT_SCHEMA_VERSION,
        "logical_schema": {
            "id": KEYPOINT_SCHEMA_V2.schema_id,
            "version": KEYPOINT_SCHEMA_V2.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "source_crop_manifest_digest": crop_source.manifest_digest,
        "pose_model_schema_binding_digest": binding["binding_sha256"],
        "skeleton_digest": skeleton_digest,
        "preprocessing_digest": preprocessing.preprocessing_digest,
        "arrays": declarations,
    }


def keypoint_metadata_declarations_digest(
    direct_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_by_path: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = {"", *KEYPOINT_SCHEMA_V2.binding_paths}
    if set(direct_by_path) != expected or set(consolidated_by_path) != expected:
        raise ValueError("Keypoint metadata declaration paths are incomplete.")
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
                    raise ValueError("Keypoint run attributes must be an object.")
                redacted = dict(attributes)
                redacted.pop(KEYPOINT_RUN_MANIFEST_ATTRIBUTE, None)
                declaration["attributes"] = redacted
        if direct != consolidated:
            raise ValueError(
                f"Direct and consolidated metadata differ at {path or '<run>'!r}."
            )
        normalized[path] = direct
    return canonical_json_sha256(
        {"scope": KEYPOINT_METADATA_DIGEST_SCOPE, "declarations": normalized}
    )


def build_keypoint_run_manifest(
    *,
    run_id: str,
    dimensions: KeypointDimensions,
    crop_source: KeypointCropSourceReference,
    source_crop_manifest: Mapping[str, Any],
    source_crop_arrays: Mapping[str, Any],
    pose_binding: Mapping[str, Any],
    preprocessing: KeypointPreprocessingReference,
    storage_plan: KeypointStoragePlanSet,
    arrays: Mapping[str, Any],
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    resolved_run_id = _require_run_id(run_id)
    if keypoint_crop_source_from_manifest(source_crop_manifest) != crop_source:
        raise ValueError("Crop manifest differs from the keypoint source binding.")
    if storage_plan.dimensions != dimensions:
        raise ValueError("Keypoint storage-plan dimensions differ.")
    binding = _validated_pose_binding(pose_binding)
    content = keypoint_logical_content_document(
        arrays,
        dimensions=dimensions,
        crop_source=crop_source,
        source_crop_arrays=source_crop_arrays,
        pose_binding=binding,
        preprocessing=preprocessing,
    )
    payload: dict[str, object] = {
        "run_id": resolved_run_id,
        "stage": "keypoints",
        "publication": {
            "artifact_class": "raw_keypoint_observations",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "keypoint_authority": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": KEYPOINT_METADATA_DIGEST_SCOPE,
            "metadata_declarations_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "metadata_declarations_digest": keypoint_metadata_declarations_digest(
                direct_metadata_declarations,
                consolidated_by_path=consolidated_metadata_declarations,
            ),
        },
        "logical_schema": KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions),
        "coordinate_contract": build_coordinate_catalog_envelope(
            KEYPOINT_SCHEMA_V2.coordinate_contract_manifest()
        ),
        "storage_plan": storage_plan.as_manifest(),
        "source_crop_snapshot": crop_source.as_manifest(),
        "pose_model_schema_binding": binding,
        "preprocessing": preprocessing.as_manifest(),
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(content),
            "document": content,
        },
    }
    envelope = {
        "schema_id": KEYPOINT_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": KEYPOINT_RUN_MANIFEST_SCHEMA_VERSION,
        "persisted_attribute": KEYPOINT_RUN_MANIFEST_ATTRIBUTE,
        "persisted_path": KEYPOINT_RUN_MANIFEST_PERSISTED_PATH,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def _parse_manifest(manifest: Mapping[str, Any]):  # type: ignore[no-untyped-def]
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
        errors.append("keypoint manifest envelope has unexpected fields")
    if (
        manifest.get("schema_id") != KEYPOINT_RUN_MANIFEST_SCHEMA_ID
        or manifest.get("schema_version") != KEYPOINT_RUN_MANIFEST_SCHEMA_VERSION
        or manifest.get("persisted_attribute") != KEYPOINT_RUN_MANIFEST_ATTRIBUTE
        or manifest.get("persisted_path") != KEYPOINT_RUN_MANIFEST_PERSISTED_PATH
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("keypoint manifest envelope identity mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return [*errors, "keypoint manifest payload must be an object"], None, None
    try:
        if manifest.get("payload_digest") != canonical_json_sha256(payload):
            errors.append("keypoint manifest payload_digest mismatch")
    except (TypeError, ValueError) as exc:
        errors.append(f"keypoint manifest is not strict JSON: {exc}")
    if set(payload) != {
        "run_id",
        "stage",
        "publication",
        "logical_schema",
        "coordinate_contract",
        "storage_plan",
        "source_crop_snapshot",
        "pose_model_schema_binding",
        "preprocessing",
        "logical_content",
    }:
        errors.append("keypoint manifest payload has unexpected fields")
    try:
        _require_run_id(payload.get("run_id"))
    except ValueError as exc:
        errors.append(str(exc))
    if payload.get("stage") != "keypoints":
        errors.append("keypoint manifest stage mismatch")
    dimensions = None
    logical = payload.get("logical_schema")
    try:
        if not isinstance(logical, Mapping):
            raise TypeError("Keypoint logical schema must be an object.")
        dimensions = _dimensions_from_manifest(logical.get("dimensions"))
        if dict(logical) != KEYPOINT_SCHEMA_V2.as_manifest(dimensions=dimensions):
            errors.append("keypoint logical schema differs from frozen builder")
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors, payload, dimensions


def validate_keypoint_run_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    errors, payload, dimensions = _parse_manifest(manifest)
    if payload is None:
        return tuple(errors)
    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("keypoint publication must be an object")
    else:
        expected = {
            "artifact_class": "raw_keypoint_observations",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": False,
            "keypoint_authority": False,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": KEYPOINT_METADATA_DIGEST_SCOPE,
            "metadata_declarations_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "metadata_declarations_digest": publication.get(
                "metadata_declarations_digest"
            ),
        }
        if dict(publication) != expected:
            errors.append("keypoint publication is not in exact persisted form")
        try:
            _require_sha256(
                publication.get("metadata_declarations_digest"),
                name="metadata_declarations_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))
    errors.extend(
        validate_coordinate_catalog_envelope(
            payload.get("coordinate_contract"),
            expected_document=KEYPOINT_SCHEMA_V2.coordinate_contract_manifest(),
        )
    )
    crop_source = preprocessing = binding = None
    try:
        raw_crop = payload.get("source_crop_snapshot")
        if not isinstance(raw_crop, Mapping):
            raise TypeError("Keypoint crop source must be an object.")
        crop_source = keypoint_crop_source_from_persisted(raw_crop)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    try:
        raw_preprocessing = payload.get("preprocessing")
        if not isinstance(raw_preprocessing, Mapping):
            raise TypeError("Keypoint preprocessing must be an object.")
        preprocessing = keypoint_preprocessing_from_manifest(raw_preprocessing)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    try:
        raw_binding = payload.get("pose_model_schema_binding")
        if not isinstance(raw_binding, Mapping):
            raise TypeError("Pose-model binding must be an object.")
        binding = _validated_pose_binding(raw_binding)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    if dimensions is not None and crop_source is not None:
        expected_dimensions = (
            crop_source.n_frames,
            crop_source.n_instances,
            crop_source.source_width,
            crop_source.source_height,
        )
        observed_dimensions = (
            dimensions.n_frames,
            dimensions.n_instances,
            dimensions.source_width,
            dimensions.source_height,
        )
        if observed_dimensions != expected_dimensions:
            errors.append("keypoint and crop dimensions differ")
    if dimensions is not None and binding is not None:
        if binding["pose_schema"]["kpt_shape"][0] != dimensions.n_keypoints:
            errors.append("keypoint and pose-schema cardinality differ")
    storage = payload.get("storage_plan")
    if not isinstance(storage, Mapping):
        errors.append("keypoint storage plan must be an object")
    elif dimensions is not None:
        profile = storage.get("storage_profile")
        try:
            if not isinstance(profile, Mapping):
                raise TypeError("Keypoint storage profile must be an object.")
            expected_storage = plan_keypoint_storage(
                dimensions, profile=storage_profile_from_manifest(profile)
            ).as_manifest()
            if dict(storage) != expected_storage:
                errors.append("keypoint storage plan differs from planner output")
        except (TypeError, ValueError) as exc:
            errors.append(f"cannot reconstruct keypoint storage plan: {exc}")
    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping) or set(logical_content) != {
        "digest_algorithm",
        "digest",
        "document",
    }:
        errors.append("keypoint logical content envelope is invalid")
        return tuple(errors)
    document = logical_content.get("document")
    if not isinstance(document, Mapping):
        return (*errors, "keypoint logical content document must be an object")
    if logical_content.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
        errors.append("keypoint logical content digest algorithm mismatch")
    if logical_content.get("digest") != canonical_json_sha256(document):
        errors.append("keypoint logical content digest mismatch")
    if set(document) != {
        "schema_id",
        "schema_version",
        "logical_schema",
        "dimensions",
        "source_crop_manifest_digest",
        "pose_model_schema_binding_digest",
        "skeleton_digest",
        "preprocessing_digest",
        "arrays",
    }:
        errors.append("keypoint logical content has unexpected fields")
    if (
        document.get("schema_id") != KEYPOINT_LOGICAL_CONTENT_SCHEMA_ID
        or document.get("schema_version") != KEYPOINT_LOGICAL_CONTENT_SCHEMA_VERSION
        or document.get("logical_schema")
        != {
            "id": KEYPOINT_SCHEMA_V2.schema_id,
            "version": KEYPOINT_SCHEMA_V2.schema_version,
        }
    ):
        errors.append("keypoint logical content identity mismatch")
    if (
        dimensions is not None
        and document.get("dimensions") != dimensions.as_manifest()
    ):
        errors.append("keypoint logical content dimensions mismatch")
    if (
        crop_source is not None
        and document.get("source_crop_manifest_digest") != crop_source.manifest_digest
    ):
        errors.append("keypoint logical content crop digest mismatch")
    if binding is not None:
        if (
            document.get("pose_model_schema_binding_digest")
            != binding["binding_sha256"]
        ):
            errors.append("keypoint logical content pose binding digest mismatch")
        if document.get("skeleton_digest") != keypoint_skeleton_digest(binding):
            errors.append("keypoint logical content skeleton digest mismatch")
    if (
        preprocessing is not None
        and document.get("preprocessing_digest") != preprocessing.preprocessing_digest
    ):
        errors.append("keypoint logical content preprocessing digest mismatch")
    arrays = document.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(
        KEYPOINT_SCHEMA_V2.binding_paths
    ):
        errors.append("keypoint logical content array declarations mismatch")
    elif dimensions is not None:
        bindings = {binding.path: binding for binding in KEYPOINT_SCHEMA_V2.bindings}
        for path, item in arrays.items():
            if not isinstance(item, Mapping) or set(item) != {
                "shape",
                "dtype",
                "digest_algorithm",
                "sha256",
            }:
                errors.append(f"keypoint logical declaration invalid at {path}")
                continue
            contract = KEYPOINT_SCHEMA_V2.contracts.resolve(
                bindings[path].contract_id, bindings[path].contract_version
            )
            expected_shape = [
                axis if isinstance(axis, int) else dimensions.contract_dimensions[axis]
                for axis in contract.shape_template
            ]
            if item.get("shape") != expected_shape or item.get("dtype") != str(
                contract.dtype.numpy_dtype
            ):
                errors.append(f"keypoint logical declaration mismatch at {path}")
            if item.get("digest_algorithm") != KEYPOINT_ARRAY_DIGEST_ALGORITHM:
                errors.append(f"keypoint array digest algorithm mismatch at {path}")
            try:
                _require_sha256(item.get("sha256"), name=f"{path} sha256")
            except ValueError as exc:
                errors.append(str(exc))
    return tuple(errors)


def validate_keypoint_publication(
    manifest: Mapping[str, Any],
    *,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Any],
    source_crop_arrays: Mapping[str, Any],
    source_crop_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    errors = list(validate_keypoint_run_manifest(manifest))
    _, payload, dimensions = _parse_manifest(manifest)
    if payload is None or dimensions is None:
        return (*errors, "keypoint manifest components are invalid")
    try:
        crop_source = keypoint_crop_source_from_manifest(source_crop_manifest)
        if crop_source.as_manifest() != payload["source_crop_snapshot"]:
            errors.append("keypoint source crop manifest binding mismatch")
        binding = _validated_pose_binding(payload["pose_model_schema_binding"])
        preprocessing = keypoint_preprocessing_from_manifest(payload["preprocessing"])
        content = keypoint_logical_content_document(
            arrays,
            dimensions=dimensions,
            crop_source=crop_source,
            source_crop_arrays=source_crop_arrays,
            pose_binding=binding,
            preprocessing=preprocessing,
        )
        if content != payload["logical_content"]["document"]:
            errors.append("keypoint logical content differs from decoded arrays")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"keypoint logical validation failed: {exc}")
    try:
        digest = keypoint_metadata_declarations_digest(
            direct_metadata_declarations,
            consolidated_by_path=consolidated_metadata_declarations,
        )
        if digest != payload["publication"].get("metadata_declarations_digest"):
            errors.append("keypoint metadata declaration digest mismatch")
    except (TypeError, ValueError) as exc:
        errors.append(f"keypoint metadata validation failed: {exc}")
    storage = payload.get("storage_plan")
    profile = storage.get("storage_profile") if isinstance(storage, Mapping) else None
    try:
        if not isinstance(profile, Mapping):
            raise ValueError("keypoint storage profile is missing")
        plans = plan_keypoint_storage(
            dimensions, profile=storage_profile_from_manifest(profile)
        )
        bindings = {binding.path: binding for binding in KEYPOINT_SCHEMA_V2.bindings}
        for entry in plans.entries:
            declaration = direct_metadata_declarations.get(entry.rule.path)
            if not isinstance(declaration, Mapping):
                errors.append(f"missing direct metadata at {entry.rule.path}")
                continue
            contract = KEYPOINT_SCHEMA_V2.contracts.resolve(
                bindings[entry.rule.path].contract_id,
                bindings[entry.rule.path].contract_version,
            )
            errors.extend(
                f"keypoint physical metadata at {entry.rule.path}: {error}"
                for error in validate_array_metadata_declaration_from_plan(
                    declaration,
                    contract=contract,
                    plan=entry.plan,
                    fill_value=0,
                )
            )
    except (TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct keypoint physical plan: {exc}")
    return tuple(errors)


__all__ = [
    "KEYPOINT_ARRAY_DIGEST_ALGORITHM",
    "KEYPOINT_CROP_SOURCE_SCHEMA_ID",
    "KEYPOINT_LOGICAL_CONTENT_SCHEMA_ID",
    "KEYPOINT_METADATA_DIGEST_SCOPE",
    "KEYPOINT_PREPROCESSING_SCHEMA_ID",
    "KEYPOINT_RUN_MANIFEST_ATTRIBUTE",
    "KEYPOINT_RUN_MANIFEST_PERSISTED_PATH",
    "KEYPOINT_RUN_MANIFEST_SCHEMA_ID",
    "KEYPOINT_RUN_MANIFEST_SCHEMA_VERSION",
    "KEYPOINT_SKELETON_SEMANTICS_SCHEMA_ID",
    "KeypointCropSourceReference",
    "KeypointPreprocessingReference",
    "build_keypoint_run_manifest",
    "keypoint_crop_source_from_manifest",
    "keypoint_logical_content_document",
    "keypoint_metadata_declarations_digest",
    "keypoint_skeleton_digest",
    "keypoint_skeleton_document",
    "validate_keypoint_publication",
    "validate_keypoint_run_manifest",
]
