"""Strict persisted manifest and publication gate for crop geometry v1."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any, Mapping
from uuid import UUID

import numpy as np

from fisheye.shared.row_source_signature import (
    RowSourceSignatureBatch,
    RowSourceSignatureSpec,
    build_row_source_signatures,
    load_row_source_signature_spec,
)
from fisheye.shared.zarr.array_factory import (
    validate_array_metadata_declaration_from_plan,
)
from fisheye.shared.zarr.benchmark_runtime import sha256_array
from fisheye.shared.zarr.coordinate_manifest import (
    build_coordinate_catalog_envelope,
    validate_coordinate_catalog_envelope,
)
from fisheye.shared.zarr.crop_schema import (
    CROP_GEOMETRY_SCHEMA_V1,
    CropDimensions,
    CropGeometryPolicy,
    crop_geometry_policy_from_manifest,
)
from fisheye.shared.zarr.crop_storage import (
    CropGeometryStoragePlanSet,
    plan_crop_geometry_storage,
)
from fisheye.shared.zarr.manifest_digest import (
    CANONICAL_JSON_DIGEST_ALGORITHM,
    canonical_json_bytes,
    canonical_json_sha256,
    metadata_without_empty_group_consolidation,
)
from fisheye.shared.zarr.refined_detection_manifest import (
    parse_refined_detection_clipped_binding,
    refined_detection_dimensions_from_manifest,
    refined_detection_logical_content_digest,
    validate_refined_detection_run_manifest,
)
from fisheye.shared.zarr.storage_profiles import storage_profile_from_manifest


CROP_RUN_MANIFEST_SCHEMA_ID = "palette.crop_geometry.run_manifest"
CROP_RUN_MANIFEST_SCHEMA_VERSION = 1
CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION = 2
CROP_RUN_MANIFEST_ATTRIBUTE = "run_manifest"
CROP_RUN_MANIFEST_PERSISTED_PATH = "crop_runs/<run>/zarr.json.attributes.run_manifest"
CROP_LOGICAL_CONTENT_SCHEMA_ID = "palette.crop_geometry.logical_content"
CROP_LOGICAL_CONTENT_SCHEMA_VERSION = 1
CROP_ARRAY_DIGEST_ALGORITHM = "sha256_c_contiguous_bytes_v1"
CROP_METADATA_DECLARATIONS_SCHEMA_ID = "palette.crop_geometry.metadata_declarations"
CROP_METADATA_DECLARATIONS_SCHEMA_VERSION = 1
CROP_METADATA_DIGEST_SCOPE = (
    "exact_group_and_array_declarations_with_attributes_redacting_only_run_manifest"
)
CROP_REFINED_SOURCE_SCHEMA_ID = "palette.crop_geometry.refined_source"
CROP_REFINED_SOURCE_SCHEMA_VERSION = 1
CROP_PIXEL_AUTHORITY_SCHEMA_ID = "palette.crop_geometry.pixel_authority"
CROP_PIXEL_AUTHORITY_SCHEMA_VERSION = 1
CROP_ROW_SIGNATURE_STAGE = "crop_geometry"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


def _require_text(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")
    return normalized


def _require_run_id(value: object, *, name: str = "run_id") -> str:
    normalized = _require_text(value, name=name)
    if "/" in normalized:
        raise ValueError(f"{name} must be one path-safe group name.")
    return normalized


def _require_uuid(value: object, *, name: str) -> str:
    normalized = _require_text(value, name=name)
    try:
        parsed = UUID(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be a UUID.") from exc
    canonical = str(parsed)
    if normalized != canonical:
        raise ValueError(f"{name} must use canonical lowercase UUID form.")
    return canonical


def _array_values(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value[...])
    except (IndexError, KeyError, TypeError):
        return np.asarray(value)


@dataclass(frozen=True)
class CropRefinedSourceIdentity:
    """Exact immutable refined-detection snapshot bound by a crop run."""

    run_id: str
    run_manifest_digest: str
    logical_content_digest: str
    recording_identity: str
    lineage_id: str
    snapshot_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_run_id(self.run_id))
        object.__setattr__(
            self,
            "run_manifest_digest",
            _require_sha256(
                self.run_manifest_digest,
                name="source refined run_manifest_digest",
            ),
        )
        object.__setattr__(
            self,
            "logical_content_digest",
            _require_sha256(
                self.logical_content_digest,
                name="source refined logical_content_digest",
            ),
        )
        object.__setattr__(
            self,
            "recording_identity",
            _require_text(self.recording_identity, name="recording_identity"),
        )
        object.__setattr__(
            self,
            "lineage_id",
            _require_uuid(self.lineage_id, name="source lineage_id"),
        )
        object.__setattr__(
            self,
            "snapshot_id",
            _require_uuid(self.snapshot_id, name="source snapshot_id"),
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": CROP_REFINED_SOURCE_SCHEMA_ID,
            "schema_version": CROP_REFINED_SOURCE_SCHEMA_VERSION,
            "authority_kind": "refined_detection_run",
            "stage": "refined_detect",
            "logical_schema": {
                "id": "palette.stage.refined_detection",
                "version": 1,
            },
            "row_coverage": "complete_instances_rowset",
            "run_id": self.run_id,
            "run_manifest_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "run_manifest_digest": self.run_manifest_digest,
            "logical_content_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "logical_content_digest": self.logical_content_digest,
            "recording_identity": self.recording_identity,
            "lineage_id": self.lineage_id,
            "snapshot_id": self.snapshot_id,
        }


def crop_refined_source_identity_from_manifest(
    value: Mapping[str, Any],
) -> CropRefinedSourceIdentity:
    expected_fields = {
        "schema_id",
        "schema_version",
        "authority_kind",
        "stage",
        "logical_schema",
        "row_coverage",
        "run_id",
        "run_manifest_digest_algorithm",
        "run_manifest_digest",
        "logical_content_digest_algorithm",
        "logical_content_digest",
        "recording_identity",
        "lineage_id",
        "snapshot_id",
    }
    if set(value) != expected_fields:
        raise ValueError("Crop refined source identity has an unexpected field set.")
    if (
        value.get("schema_id") != CROP_REFINED_SOURCE_SCHEMA_ID
        or value.get("schema_version") != CROP_REFINED_SOURCE_SCHEMA_VERSION
        or value.get("authority_kind") != "refined_detection_run"
        or value.get("stage") != "refined_detect"
        or value.get("row_coverage") != "complete_instances_rowset"
    ):
        raise ValueError("Crop refined source identity header mismatch.")
    if value.get("logical_schema") != {
        "id": "palette.stage.refined_detection",
        "version": 1,
    }:
        raise ValueError("Crop refined source logical schema mismatch.")
    if value.get("run_manifest_digest_algorithm") != (
        CANONICAL_JSON_DIGEST_ALGORITHM
    ) or value.get("logical_content_digest_algorithm") != (
        CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        raise ValueError("Crop refined source digest algorithm mismatch.")
    identity = CropRefinedSourceIdentity(
        run_id=value.get("run_id"),
        run_manifest_digest=value.get("run_manifest_digest"),
        logical_content_digest=value.get("logical_content_digest"),
        recording_identity=value.get("recording_identity"),
        lineage_id=value.get("lineage_id"),
        snapshot_id=value.get("snapshot_id"),
    )
    if dict(value) != identity.as_manifest():
        raise ValueError("Crop refined source identity is not canonical.")
    return identity


def crop_refined_source_identity_from_refined_manifest(
    manifest: Mapping[str, Any],
    *,
    logical_content_digest: str,
) -> CropRefinedSourceIdentity:
    errors = validate_refined_detection_run_manifest(manifest)
    if errors:
        raise ValueError("Invalid refined source manifest: " + "; ".join(errors))
    payload = manifest["payload"]
    lineage = payload["snapshot_lineage"]
    allocator = lineage["manual_instance_key_allocator"]
    return CropRefinedSourceIdentity(
        run_id=payload["run_id"],
        run_manifest_digest=manifest["payload_digest"],
        logical_content_digest=logical_content_digest,
        recording_identity=allocator["recording_identity"],
        lineage_id=lineage["lineage_id"],
        snapshot_id=lineage["snapshot_id"],
    )


@dataclass(frozen=True)
class CropPixelAuthority:
    """Exact source-pixel identity for a geometry-only crop publication."""

    authority_id: str
    authority_manifest_digest: str
    recording_identity: str
    camera_identity: str
    n_frames: int
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        for name in ("authority_id", "recording_identity", "camera_identity"):
            object.__setattr__(
                self,
                name,
                _require_text(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "authority_manifest_digest",
            _require_sha256(
                self.authority_manifest_digest,
                name="pixel authority_manifest_digest",
            ),
        )
        for name in ("n_frames", "source_width", "source_height"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Pixel authority {name} must be positive integer.")

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": CROP_PIXEL_AUTHORITY_SCHEMA_ID,
            "schema_version": CROP_PIXEL_AUTHORITY_SCHEMA_VERSION,
            "provider_profile": "source_video_geometry_v1",
            "authority_id": self.authority_id,
            "authority_manifest_digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "authority_manifest_digest": self.authority_manifest_digest,
            "recording_identity": self.recording_identity,
            "camera_identity": self.camera_identity,
            "frame_index_domain": "zero_based_acquisition_camera_frame",
            "n_frames": self.n_frames,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "decoded_pixel_contract": {
                "dtype": "uint8",
                "channels": "grayscale",
                "axis_order": "yx",
                "crop_sampling": "integer_half_open_xywh",
            },
        }


def crop_pixel_authority_from_manifest(
    value: Mapping[str, Any],
) -> CropPixelAuthority:
    expected_fields = {
        "schema_id",
        "schema_version",
        "provider_profile",
        "authority_id",
        "authority_manifest_digest_algorithm",
        "authority_manifest_digest",
        "recording_identity",
        "camera_identity",
        "frame_index_domain",
        "n_frames",
        "source_width",
        "source_height",
        "decoded_pixel_contract",
    }
    if set(value) != expected_fields:
        raise ValueError("Crop pixel authority has an unexpected field set.")
    if (
        value.get("schema_id") != CROP_PIXEL_AUTHORITY_SCHEMA_ID
        or value.get("schema_version") != CROP_PIXEL_AUTHORITY_SCHEMA_VERSION
        or value.get("provider_profile") != "source_video_geometry_v1"
        or value.get("authority_manifest_digest_algorithm")
        != CANONICAL_JSON_DIGEST_ALGORITHM
        or value.get("frame_index_domain") != "zero_based_acquisition_camera_frame"
    ):
        raise ValueError("Crop pixel authority header mismatch.")
    authority = CropPixelAuthority(
        authority_id=value.get("authority_id"),
        authority_manifest_digest=value.get("authority_manifest_digest"),
        recording_identity=value.get("recording_identity"),
        camera_identity=value.get("camera_identity"),
        n_frames=value.get("n_frames"),
        source_width=value.get("source_width"),
        source_height=value.get("source_height"),
    )
    if dict(value) != authority.as_manifest():
        raise ValueError("Crop pixel authority is not canonical.")
    return authority


_CROP_SIGNATURE_CONTENT_PATHS = (
    "source_refined_row_ids",
    "frame_indices",
    "bbox_norm_coords",
    "roi_coordinates_full",
    "roi_sizes_full",
)


def build_crop_row_source_signatures(
    arrays: Mapping[str, Any],
    *,
    source: CropRefinedSourceIdentity,
    policy: CropGeometryPolicy,
    pixel_authority: CropPixelAuthority,
) -> RowSourceSignatureBatch:
    """Build the exact per-row reuse signature for crop geometry v1."""

    missing = {"instance_key", *_CROP_SIGNATURE_CONTENT_PATHS} - set(arrays)
    if missing:
        raise ValueError(f"Crop row signatures lack arrays: {sorted(missing)!r}.")
    return build_row_source_signatures(
        stage=CROP_ROW_SIGNATURE_STAGE,
        instance_keys=_array_values(arrays["instance_key"]),
        content_components={
            path: _array_values(arrays[path]) for path in _CROP_SIGNATURE_CONTENT_PATHS
        },
        compatibility_context={
            "crop_schema": {
                "id": CROP_GEOMETRY_SCHEMA_V1.schema_id,
                "version": CROP_GEOMETRY_SCHEMA_V1.schema_version,
            },
            "crop_policy_digest": policy.payload_digest,
            "recording_identity": source.recording_identity,
            "source_refined_run_id": source.run_id,
            "source_refined_manifest_digest": source.run_manifest_digest,
            "source_refined_logical_content_digest": (source.logical_content_digest),
            "source_refined_lineage_id": source.lineage_id,
            "source_refined_snapshot_id": source.snapshot_id,
            "source_pixel_authority_id": pixel_authority.authority_id,
            "source_pixel_authority_manifest_digest": (
                pixel_authority.authority_manifest_digest
            ),
        },
    )


def crop_row_signature_manifest(spec: RowSourceSignatureSpec) -> dict[str, object]:
    return {
        "array_path": "source_row_signature",
        **spec.to_attrs(prefix=""),
    }


def _empty_signature_arrays() -> dict[str, np.ndarray]:
    return {
        "instance_key": np.empty(0, dtype=np.uint64),
        "source_refined_row_ids": np.empty(0, dtype=np.int64),
        "frame_indices": np.empty(0, dtype=np.int64),
        "bbox_norm_coords": np.empty((0, 4), dtype=np.float32),
        "roi_coordinates_full": np.empty((0, 2), dtype=np.int32),
        "roi_sizes_full": np.empty((0, 2), dtype=np.int32),
    }


def expected_crop_row_signature_manifest(
    *,
    source: CropRefinedSourceIdentity,
    policy: CropGeometryPolicy,
    pixel_authority: CropPixelAuthority,
) -> dict[str, object]:
    batch = build_crop_row_source_signatures(
        _empty_signature_arrays(),
        source=source,
        policy=policy,
        pixel_authority=pixel_authority,
    )
    return crop_row_signature_manifest(batch.spec)


def normalize_crop_metadata_declarations(
    direct_metadata_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_metadata_by_path: Mapping[str, Mapping[str, Any]],
    dimensions: CropDimensions,
) -> dict[str, object]:
    """Normalize exact direct/consolidated declarations, retaining attributes."""

    expected_paths = {"", *CROP_GEOMETRY_SCHEMA_V1.binding_paths}
    if set(direct_metadata_by_path) != expected_paths:
        raise ValueError("Crop direct metadata declaration paths must be exact.")
    if set(consolidated_metadata_by_path) != expected_paths:
        raise ValueError("Crop consolidated metadata declaration paths must be exact.")

    direct: dict[str, Mapping[str, Any]] = {}
    for path in sorted(expected_paths):
        item = direct_metadata_by_path[path]
        candidate = consolidated_metadata_by_path[path]
        if not isinstance(item, Mapping) or not isinstance(candidate, Mapping):
            raise TypeError(f"Crop metadata declaration {path!r} must be an object.")
        canonical_json_bytes(item)
        canonical_json_bytes(candidate)
        if metadata_without_empty_group_consolidation(
            item,
            path=path,
        ) != metadata_without_empty_group_consolidation(candidate, path=path):
            raise ValueError(
                f"Direct and consolidated crop metadata differ at {path!r}."
            )
        direct[path] = item

    normalized: dict[str, dict[str, Any]] = {}
    for path in sorted(expected_paths):
        declaration = copy.deepcopy(dict(direct[path]))
        if declaration.get("zarr_format") != 3:
            raise ValueError(f"Crop declaration {path!r} must use zarr_format 3.")
        attributes = declaration.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError(f"Crop declaration {path!r} requires object attributes.")
        if path == "":
            required = {"zarr_format", "node_type", "attributes"}
            optional = {"consolidated_metadata"}
            if not required.issubset(declaration) or not set(declaration).issubset(
                required | optional
            ):
                raise ValueError("Crop run group has an unexpected field set.")
            if declaration.get("node_type") != "group":
                raise ValueError("Crop run root declaration must be a group.")
            redacted = dict(attributes)
            redacted.pop(CROP_RUN_MANIFEST_ATTRIBUTE, None)
            declaration["attributes"] = redacted
        else:
            required = {
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
            optional = {"dimension_names"}
            if not required.issubset(declaration) or not set(declaration).issubset(
                required | optional
            ):
                raise ValueError(
                    f"Crop array declaration {path!r} has an unexpected field set."
                )
            if declaration.get("node_type") != "array":
                raise ValueError(f"Crop declaration {path!r} must be an array.")
        declaration.pop("consolidated_metadata", None)
        normalized[path] = declaration

    document: dict[str, object] = {
        "schema_id": CROP_METADATA_DECLARATIONS_SCHEMA_ID,
        "schema_version": CROP_METADATA_DECLARATIONS_SCHEMA_VERSION,
        "path_basis": "relative_to_crop_run_group_empty_string_is_root",
        "included_nodes": "exact_run_root_and_schema_arrays",
        "attribute_policy": (
            "all_attributes_included_except_root_run_manifest_circular_field"
        ),
        "excluded_fields": ["consolidated_metadata", "root.attributes.run_manifest"],
        "dimensions": dimensions.as_manifest(),
        "declarations": normalized,
    }
    canonical_json_bytes(document)
    return document


def crop_metadata_declarations_digest(
    direct_metadata_by_path: Mapping[str, Mapping[str, Any]],
    *,
    consolidated_metadata_by_path: Mapping[str, Mapping[str, Any]],
    dimensions: CropDimensions,
) -> str:
    return canonical_json_sha256(
        normalize_crop_metadata_declarations(
            direct_metadata_by_path,
            consolidated_metadata_by_path=consolidated_metadata_by_path,
            dimensions=dimensions,
        )
    )


def crop_logical_content_document(
    arrays: Mapping[str, Any],
    *,
    dimensions: CropDimensions,
    policy: CropGeometryPolicy,
    source: CropRefinedSourceIdentity,
    pixel_authority: CropPixelAuthority,
) -> dict[str, object]:
    """Validate and digest every decoded crop array and its signature semantics."""

    CROP_GEOMETRY_SCHEMA_V1.require(
        arrays,
        dimensions=dimensions,
        policy=policy,
    )
    if set(arrays) != set(CROP_GEOMETRY_SCHEMA_V1.binding_paths):
        raise ValueError("Crop logical content requires the exact array set.")
    expected_signatures = build_crop_row_source_signatures(
        arrays,
        source=source,
        policy=policy,
        pixel_authority=pixel_authority,
    )
    if not np.array_equal(
        _array_values(arrays["source_row_signature"]),
        expected_signatures.signatures,
    ):
        raise ValueError("source_row_signature differs from the frozen crop spec.")

    declarations: dict[str, object] = {}
    for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths:
        values = np.ascontiguousarray(_array_values(arrays[path]))
        declarations[path] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "digest_algorithm": CROP_ARRAY_DIGEST_ALGORITHM,
            "sha256": sha256_array(values),
        }
    document: dict[str, object] = {
        "schema_id": CROP_LOGICAL_CONTENT_SCHEMA_ID,
        "schema_version": CROP_LOGICAL_CONTENT_SCHEMA_VERSION,
        "logical_schema": {
            "id": CROP_GEOMETRY_SCHEMA_V1.schema_id,
            "version": CROP_GEOMETRY_SCHEMA_V1.schema_version,
        },
        "dimensions": dimensions.as_manifest(),
        "crop_policy_digest": policy.payload_digest,
        "source_refined_manifest_digest": source.run_manifest_digest,
        "source_pixel_authority_manifest_digest": (
            pixel_authority.authority_manifest_digest
        ),
        "arrays": declarations,
    }
    canonical_json_bytes(document)
    return document


def crop_logical_content_digest(
    arrays: Mapping[str, Any],
    *,
    dimensions: CropDimensions,
    policy: CropGeometryPolicy,
    source: CropRefinedSourceIdentity,
    pixel_authority: CropPixelAuthority,
) -> str:
    return canonical_json_sha256(
        crop_logical_content_document(
            arrays,
            dimensions=dimensions,
            policy=policy,
            source=source,
            pixel_authority=pixel_authority,
        )
    )


def _require_bound_dimensions(
    dimensions: CropDimensions,
    *,
    source: CropRefinedSourceIdentity,
    pixel_authority: CropPixelAuthority,
) -> None:
    if source.recording_identity != pixel_authority.recording_identity:
        raise ValueError("Refined and pixel authorities bind different recordings.")
    if (
        dimensions.n_frames != pixel_authority.n_frames
        or dimensions.source_width != pixel_authority.source_width
        or dimensions.source_height != pixel_authority.source_height
    ):
        raise ValueError("Pixel authority dimensions differ from crop dimensions.")


def _build_crop_run_manifest(
    *,
    run_id: str,
    dimensions: CropDimensions,
    policy: CropGeometryPolicy,
    storage_plan: CropGeometryStoragePlanSet,
    arrays: Mapping[str, Any],
    source: CropRefinedSourceIdentity,
    pixel_authority: CropPixelAuthority,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    selector_eligible: bool,
    manifest_schema_version: int,
) -> dict[str, object]:
    """Build the exact persisted crop run-manifest envelope."""

    resolved_run_id = _require_run_id(run_id)
    if type(selector_eligible) is not bool:
        raise TypeError("selector_eligible must be an exact boolean.")
    if storage_plan.dimensions != dimensions:
        raise ValueError("Crop storage-plan dimensions do not match.")
    _require_bound_dimensions(
        dimensions,
        source=source,
        pixel_authority=pixel_authority,
    )
    content = crop_logical_content_document(
        arrays,
        dimensions=dimensions,
        policy=policy,
        source=source,
        pixel_authority=pixel_authority,
    )
    signature_batch = build_crop_row_source_signatures(
        arrays,
        source=source,
        policy=policy,
        pixel_authority=pixel_authority,
    )
    metadata_digest = crop_metadata_declarations_digest(
        direct_metadata_declarations,
        consolidated_metadata_by_path=consolidated_metadata_declarations,
        dimensions=dimensions,
    )
    payload: dict[str, object] = {
        "run_id": resolved_run_id,
        "stage": "crop",
        "publication": {
            "artifact_class": "geometry_only_analysis",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": selector_eligible,
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": CROP_METADATA_DIGEST_SCOPE,
            "metadata_declarations_digest_algorithm": (CANONICAL_JSON_DIGEST_ALGORITHM),
            "metadata_declarations_digest": metadata_digest,
        },
        "logical_schema": CROP_GEOMETRY_SCHEMA_V1.as_manifest(
            dimensions=dimensions,
            policy=policy,
        ),
        "storage_plan": storage_plan.as_manifest(),
        "source_refined_snapshot": source.as_manifest(),
        "source_pixel_authority": pixel_authority.as_manifest(),
        "row_signature": crop_row_signature_manifest(signature_batch.spec),
        "logical_content": {
            "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
            "digest": canonical_json_sha256(content),
            "document": content,
        },
    }
    if manifest_schema_version == CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION:
        payload["coordinate_contract"] = build_coordinate_catalog_envelope(
            CROP_GEOMETRY_SCHEMA_V1.coordinate_contract_manifest()
        )
    elif manifest_schema_version != CROP_RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported crop run-manifest version.")
    envelope: dict[str, object] = {
        "schema_id": CROP_RUN_MANIFEST_SCHEMA_ID,
        "schema_version": manifest_schema_version,
        "persisted_attribute": CROP_RUN_MANIFEST_ATTRIBUTE,
        "persisted_path": CROP_RUN_MANIFEST_PERSISTED_PATH,
        "digest_algorithm": CANONICAL_JSON_DIGEST_ALGORITHM,
        "payload_digest": canonical_json_sha256(payload),
        "payload": payload,
    }
    canonical_json_bytes(envelope)
    return envelope


def build_crop_run_manifest(
    *,
    run_id: str,
    dimensions: CropDimensions,
    policy: CropGeometryPolicy,
    storage_plan: CropGeometryStoragePlanSet,
    arrays: Mapping[str, Any],
    source: CropRefinedSourceIdentity,
    pixel_authority: CropPixelAuthority,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    selector_eligible: bool,
) -> dict[str, object]:
    """Build the unchanged crop run-manifest v1 envelope."""

    return _build_crop_run_manifest(
        run_id=run_id,
        dimensions=dimensions,
        policy=policy,
        storage_plan=storage_plan,
        arrays=arrays,
        source=source,
        pixel_authority=pixel_authority,
        direct_metadata_declarations=direct_metadata_declarations,
        consolidated_metadata_declarations=consolidated_metadata_declarations,
        selector_eligible=selector_eligible,
        manifest_schema_version=CROP_RUN_MANIFEST_SCHEMA_VERSION,
    )


def build_coordinate_crop_run_manifest(
    *,
    run_id: str,
    dimensions: CropDimensions,
    policy: CropGeometryPolicy,
    storage_plan: CropGeometryStoragePlanSet,
    arrays: Mapping[str, Any],
    source: CropRefinedSourceIdentity,
    pixel_authority: CropPixelAuthority,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    selector_eligible: bool,
) -> dict[str, object]:
    """Build opt-in v2 with an exact persisted coordinate catalog."""

    return _build_crop_run_manifest(
        run_id=run_id,
        dimensions=dimensions,
        policy=policy,
        storage_plan=storage_plan,
        arrays=arrays,
        source=source,
        pixel_authority=pixel_authority,
        direct_metadata_declarations=direct_metadata_declarations,
        consolidated_metadata_declarations=consolidated_metadata_declarations,
        selector_eligible=selector_eligible,
        manifest_schema_version=CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    )


def _dimensions_and_policy_from_logical(
    logical: Mapping[str, Any],
) -> tuple[CropDimensions, CropGeometryPolicy]:
    raw = logical.get("dimensions")
    if not isinstance(raw, Mapping) or set(raw) != {
        "n_frames",
        "n_instances",
        "n_frame_boundaries",
        "source_width",
        "source_height",
    }:
        raise ValueError("Crop logical dimensions are not exact.")
    dimensions = CropDimensions(
        n_frames=raw.get("n_frames"),
        n_instances=raw.get("n_instances"),
        source_width=raw.get("source_width"),
        source_height=raw.get("source_height"),
    )
    if raw.get("n_frame_boundaries") != dimensions.n_frames + 1:
        raise ValueError("Crop n_frame_boundaries must equal n_frames + 1.")
    policy_value = logical.get("crop_policy")
    if not isinstance(policy_value, Mapping):
        raise TypeError("Crop logical schema lacks crop_policy.")
    policy = crop_geometry_policy_from_manifest(policy_value)
    return dimensions, policy


def _validate_logical_content_declarations(
    content: Mapping[str, Any],
    *,
    dimensions: CropDimensions,
    policy: CropGeometryPolicy,
    source: CropRefinedSourceIdentity,
    pixel_authority: CropPixelAuthority,
) -> tuple[str, ...]:
    errors: list[str] = []
    expected_fields = {
        "schema_id",
        "schema_version",
        "logical_schema",
        "dimensions",
        "crop_policy_digest",
        "source_refined_manifest_digest",
        "source_pixel_authority_manifest_digest",
        "arrays",
    }
    if set(content) != expected_fields:
        errors.append("crop logical_content document has unexpected fields")
    if (
        content.get("schema_id") != CROP_LOGICAL_CONTENT_SCHEMA_ID
        or content.get("schema_version") != CROP_LOGICAL_CONTENT_SCHEMA_VERSION
    ):
        errors.append("crop logical_content schema identity mismatch")
    if content.get("logical_schema") != {
        "id": CROP_GEOMETRY_SCHEMA_V1.schema_id,
        "version": CROP_GEOMETRY_SCHEMA_V1.schema_version,
    }:
        errors.append("crop logical_content logical schema mismatch")
    if content.get("dimensions") != dimensions.as_manifest():
        errors.append("crop logical_content dimensions mismatch")
    if content.get("crop_policy_digest") != policy.payload_digest:
        errors.append("crop logical_content policy digest mismatch")
    if content.get("source_refined_manifest_digest") != source.run_manifest_digest:
        errors.append("crop logical_content refined source digest mismatch")
    if content.get("source_pixel_authority_manifest_digest") != (
        pixel_authority.authority_manifest_digest
    ):
        errors.append("crop logical_content pixel authority digest mismatch")

    arrays = content.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(
        CROP_GEOMETRY_SCHEMA_V1.binding_paths
    ):
        return (*errors, "crop logical_content array declarations mismatch")
    bindings = {binding.path: binding for binding in CROP_GEOMETRY_SCHEMA_V1.bindings}
    for path in CROP_GEOMETRY_SCHEMA_V1.binding_paths:
        item = arrays[path]
        if not isinstance(item, Mapping) or set(item) != {
            "shape",
            "dtype",
            "digest_algorithm",
            "sha256",
        }:
            errors.append(f"crop logical_content declaration invalid at {path!r}")
            continue
        binding = bindings[path]
        contract = CROP_GEOMETRY_SCHEMA_V1.contracts.resolve(
            binding.contract_id,
            binding.contract_version,
        )
        expected_shape = [
            (
                dimension
                if isinstance(dimension, int)
                else dimensions.contract_dimensions[dimension]
            )
            for dimension in contract.shape_template
        ]
        if item.get("shape") != expected_shape:
            errors.append(f"crop logical_content shape mismatch at {path!r}")
        if item.get("dtype") != str(contract.dtype.numpy_dtype):
            errors.append(f"crop logical_content dtype mismatch at {path!r}")
        if item.get("digest_algorithm") != CROP_ARRAY_DIGEST_ALGORITHM:
            errors.append(f"crop logical_content digest algorithm mismatch at {path!r}")
        try:
            _require_sha256(item.get("sha256"), name=f"logical_content {path} sha256")
        except ValueError as exc:
            errors.append(str(exc))
    return tuple(errors)


def validate_crop_run_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Deeply validate the complete persisted envelope without opening arrays."""

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
        errors.append("crop run manifest envelope has unexpected fields")
    manifest_schema_version = manifest.get("schema_version")
    if (
        manifest.get("schema_id") != CROP_RUN_MANIFEST_SCHEMA_ID
        or manifest_schema_version
        not in {
            CROP_RUN_MANIFEST_SCHEMA_VERSION,
            CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
        }
        or manifest.get("persisted_attribute") != CROP_RUN_MANIFEST_ATTRIBUTE
        or manifest.get("persisted_path") != CROP_RUN_MANIFEST_PERSISTED_PATH
        or manifest.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM
    ):
        errors.append("crop run manifest envelope identity mismatch")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        return (*errors, "crop run manifest payload must be an object")
    try:
        expected_digest = canonical_json_sha256(payload)
    except (TypeError, ValueError) as exc:
        return (*errors, f"crop run manifest is not strict JSON: {exc}")
    if manifest.get("payload_digest") != expected_digest:
        errors.append("crop run manifest payload_digest mismatch")
    expected_payload_fields = {
        "run_id",
        "stage",
        "publication",
        "logical_schema",
        "storage_plan",
        "source_refined_snapshot",
        "source_pixel_authority",
        "row_signature",
        "logical_content",
    }
    if manifest_schema_version == CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION:
        expected_payload_fields.add("coordinate_contract")
    if set(payload) != expected_payload_fields:
        errors.append("crop run manifest payload has unexpected fields")
    try:
        _require_run_id(payload.get("run_id"))
    except ValueError as exc:
        errors.append(str(exc))
    if payload.get("stage") != "crop":
        errors.append("crop run manifest stage mismatch")

    publication = payload.get("publication")
    if not isinstance(publication, Mapping):
        errors.append("crop publication must be an object")
    else:
        expected_publication = {
            "artifact_class": "geometry_only_analysis",
            "completion_contract": "palette.zarr_run_completion.v1",
            "completion_status": "complete",
            "stage_selector_eligible": publication.get("stage_selector_eligible"),
            "metadata_state": "direct_and_consolidated_validated",
            "metadata_declarations_digest_scope": CROP_METADATA_DIGEST_SCOPE,
            "metadata_declarations_digest_algorithm": (CANONICAL_JSON_DIGEST_ALGORITHM),
            "metadata_declarations_digest": publication.get(
                "metadata_declarations_digest"
            ),
        }
        if dict(publication) != expected_publication:
            errors.append("crop publication is not in exact persisted form")
        if type(publication.get("stage_selector_eligible")) is not bool:
            errors.append("crop selector eligibility must be boolean")
        try:
            _require_sha256(
                publication.get("metadata_declarations_digest"),
                name="metadata_declarations_digest",
            )
        except ValueError as exc:
            errors.append(str(exc))

    logical = payload.get("logical_schema")
    dimensions: CropDimensions | None = None
    policy: CropGeometryPolicy | None = None
    if not isinstance(logical, Mapping):
        errors.append("crop logical_schema must be an object")
    else:
        try:
            dimensions, policy = _dimensions_and_policy_from_logical(logical)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if dict(logical) != CROP_GEOMETRY_SCHEMA_V1.as_manifest(
                dimensions=dimensions,
                policy=policy,
            ):
                errors.append("crop logical_schema differs from frozen builder")

    if manifest_schema_version == CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION:
        errors.extend(
            validate_coordinate_catalog_envelope(
                payload.get("coordinate_contract"),
                expected_document=CROP_GEOMETRY_SCHEMA_V1.coordinate_contract_manifest(),
            )
        )

    storage = payload.get("storage_plan")
    if not isinstance(storage, Mapping):
        errors.append("crop storage_plan must be an object")
    elif dimensions is not None:
        raw_profile = storage.get("storage_profile")
        if not isinstance(raw_profile, Mapping):
            errors.append("crop storage_plan storage_profile must be an object")
        else:
            try:
                profile = storage_profile_from_manifest(raw_profile)
                expected_storage = plan_crop_geometry_storage(
                    dimensions,
                    profile=profile,
                ).as_manifest()
            except (TypeError, ValueError) as exc:
                errors.append(f"cannot reconstruct crop storage_plan: {exc}")
            else:
                if dict(storage) != expected_storage:
                    errors.append("crop storage_plan differs from byte planner output")

    source: CropRefinedSourceIdentity | None = None
    source_value = payload.get("source_refined_snapshot")
    if not isinstance(source_value, Mapping):
        errors.append("crop source_refined_snapshot must be an object")
    else:
        try:
            source = crop_refined_source_identity_from_manifest(source_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    pixel: CropPixelAuthority | None = None
    pixel_value = payload.get("source_pixel_authority")
    if not isinstance(pixel_value, Mapping):
        errors.append("crop source_pixel_authority must be an object")
    else:
        try:
            pixel = crop_pixel_authority_from_manifest(pixel_value)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    if dimensions is not None and source is not None and pixel is not None:
        try:
            _require_bound_dimensions(
                dimensions,
                source=source,
                pixel_authority=pixel,
            )
        except ValueError as exc:
            errors.append(str(exc))

    row_signature = payload.get("row_signature")
    if not isinstance(row_signature, Mapping):
        errors.append("crop row_signature must be an object")
    elif source is not None and policy is not None and pixel is not None:
        expected_row_signature = expected_crop_row_signature_manifest(
            source=source,
            policy=policy,
            pixel_authority=pixel,
        )
        if dict(row_signature) != expected_row_signature:
            errors.append("crop row_signature differs from frozen builder")
        else:
            try:
                load_row_source_signature_spec(
                    {
                        key: value
                        for key, value in row_signature.items()
                        if key != "array_path"
                    },
                    prefix="",
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"crop row_signature is invalid: {exc}")

    logical_content = payload.get("logical_content")
    if not isinstance(logical_content, Mapping) or set(logical_content) != {
        "digest_algorithm",
        "digest",
        "document",
    }:
        errors.append("crop logical_content envelope is invalid")
    else:
        if logical_content.get("digest_algorithm") != CANONICAL_JSON_DIGEST_ALGORITHM:
            errors.append("crop logical_content digest algorithm mismatch")
        document = logical_content.get("document")
        if not isinstance(document, Mapping):
            errors.append("crop logical_content document is invalid")
        elif logical_content.get("digest") != canonical_json_sha256(document):
            errors.append("crop logical_content digest mismatch")
        if (
            isinstance(document, Mapping)
            and dimensions is not None
            and policy is not None
            and source is not None
            and pixel is not None
        ):
            errors.extend(
                _validate_logical_content_declarations(
                    document,
                    dimensions=dimensions,
                    policy=policy,
                    source=source,
                    pixel_authority=pixel,
                )
            )
    return tuple(dict.fromkeys(errors))


def validate_crop_publication(
    manifest: Mapping[str, Any],
    *,
    direct_metadata_declarations: Mapping[str, Mapping[str, Any]],
    consolidated_metadata_declarations: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recompute crop, metadata, physical, and refined-source evidence."""

    errors = list(validate_crop_run_manifest(manifest))
    payload = manifest.get("payload")
    logical = payload.get("logical_schema") if isinstance(payload, Mapping) else None
    if not isinstance(logical, Mapping):
        return (*errors, "crop publication lacks logical_schema")
    try:
        dimensions, policy = _dimensions_and_policy_from_logical(logical)
        source = crop_refined_source_identity_from_manifest(
            payload["source_refined_snapshot"]
        )
        pixel = crop_pixel_authority_from_manifest(payload["source_pixel_authority"])
    except (KeyError, TypeError, ValueError) as exc:
        return (*errors, f"crop publication bindings are invalid: {exc}")

    try:
        observed_content = crop_logical_content_document(
            arrays,
            dimensions=dimensions,
            policy=policy,
            source=source,
            pixel_authority=pixel,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"crop logical array validation failed: {exc}")
    else:
        expected_content = payload["logical_content"]["document"]
        if observed_content != expected_content:
            errors.append("crop logical_content differs from decoded arrays")

    try:
        metadata_digest = crop_metadata_declarations_digest(
            direct_metadata_declarations,
            consolidated_metadata_by_path=consolidated_metadata_declarations,
            dimensions=dimensions,
        )
    except (TypeError, ValueError) as exc:
        errors.append(f"crop metadata declaration validation failed: {exc}")
    else:
        if metadata_digest != payload["publication"].get(
            "metadata_declarations_digest"
        ):
            errors.append("crop metadata declaration digest mismatch")

    storage = payload.get("storage_plan")
    raw_profile = (
        storage.get("storage_profile") if isinstance(storage, Mapping) else None
    )
    try:
        if not isinstance(raw_profile, Mapping):
            raise ValueError("crop storage profile is missing")
        profile = storage_profile_from_manifest(raw_profile)
        plans = plan_crop_geometry_storage(dimensions, profile=profile)
    except (TypeError, ValueError) as exc:
        errors.append(f"cannot reconstruct crop physical plan: {exc}")
    else:
        bindings = {
            binding.path: binding for binding in CROP_GEOMETRY_SCHEMA_V1.bindings
        }
        for entry in plans.entries:
            declaration = direct_metadata_declarations.get(entry.rule.path)
            if not isinstance(declaration, Mapping):
                continue
            binding = bindings[entry.rule.path]
            contract = CROP_GEOMETRY_SCHEMA_V1.contracts.resolve(
                binding.contract_id,
                binding.contract_version,
            )
            physical_errors = validate_array_metadata_declaration_from_plan(
                declaration,
                contract=contract,
                plan=entry.plan,
                fill_value=0,
            )
            errors.extend(
                f"crop physical metadata at {entry.rule.path}: {error}"
                for error in physical_errors
            )

    source_errors = validate_refined_detection_run_manifest(source_manifest)
    errors.extend(f"source refined manifest: {error}" for error in source_errors)
    if not source_errors:
        source_payload = source_manifest["payload"]
        source_lineage = source_payload["snapshot_lineage"]
        source_allocator = source_lineage["manual_instance_key_allocator"]
        if source_payload["run_id"] != source.run_id:
            errors.append("source refined run_id differs from crop binding")
        if source_manifest["payload_digest"] != source.run_manifest_digest:
            errors.append("source refined manifest digest differs from crop binding")
        if source_lineage["lineage_id"] != source.lineage_id:
            errors.append("source refined lineage_id differs from crop binding")
        if source_lineage["snapshot_id"] != source.snapshot_id:
            errors.append("source refined snapshot_id differs from crop binding")
        if source_allocator["recording_identity"] != source.recording_identity:
            errors.append("source refined recording identity differs from crop binding")
        try:
            source_dimensions = refined_detection_dimensions_from_manifest(
                source_manifest
            )
            raw_clipped_binding = source_payload["logical_schema"].get(
                "clipped_binding"
            )
            clipped_binding = (
                None
                if raw_clipped_binding is None
                else parse_refined_detection_clipped_binding(raw_clipped_binding)
            )
            observed_source_digest = refined_detection_logical_content_digest(
                source_arrays,
                dimensions=source_dimensions,
                clipped_binding=clipped_binding,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"source refined decoded arrays are invalid: {exc}")
        else:
            if observed_source_digest != source.logical_content_digest:
                errors.append("source refined logical digest differs from crop binding")
            if (
                source_dimensions.n_frames != dimensions.n_frames
                or source_dimensions.n_instances != dimensions.n_instances
                or source_dimensions.source_width != dimensions.source_width
                or source_dimensions.source_height != dimensions.source_height
            ):
                errors.append("source refined dimensions differ from crop dimensions")
            comparisons = (
                ("instance_key", "instances/instance_key", None),
                ("source_refined_row_ids", "instances/refined_row_ids", None),
                ("frame_indices", "instances/frame_indices", np.int64),
                (
                    "source_acquisition_frame_index",
                    "instances/source_acquisition_frame_index",
                    None,
                ),
                ("frame_row_offsets", "instances/frame_row_offsets", None),
                ("bbox_norm_coords", "instances/bbox_norm_coords", None),
                ("bbox_img_xyxy", "instances/bbox_img_xyxy", None),
                ("centers_img_xy", "instances/centers_img_xy", None),
            )
            for crop_path, refined_path, dtype in comparisons:
                if crop_path not in arrays or refined_path not in source_arrays:
                    errors.append(
                        f"crop/source comparison lacks {crop_path!r} or {refined_path!r}"
                    )
                    continue
                refined_values = _array_values(source_arrays[refined_path])
                if dtype is not None:
                    refined_values = refined_values.astype(dtype)
                if not np.array_equal(
                    _array_values(arrays[crop_path]),
                    refined_values,
                ):
                    errors.append(
                        f"crop array {crop_path!r} differs from bound refined rows"
                    )
    return tuple(dict.fromkeys(errors))


__all__ = [
    "CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION",
    "CROP_ARRAY_DIGEST_ALGORITHM",
    "CROP_METADATA_DIGEST_SCOPE",
    "CROP_RUN_MANIFEST_ATTRIBUTE",
    "CROP_RUN_MANIFEST_PERSISTED_PATH",
    "CROP_RUN_MANIFEST_SCHEMA_ID",
    "CROP_RUN_MANIFEST_SCHEMA_VERSION",
    "CropPixelAuthority",
    "CropRefinedSourceIdentity",
    "build_crop_row_source_signatures",
    "build_crop_run_manifest",
    "build_coordinate_crop_run_manifest",
    "crop_logical_content_digest",
    "crop_logical_content_document",
    "crop_metadata_declarations_digest",
    "crop_pixel_authority_from_manifest",
    "crop_refined_source_identity_from_manifest",
    "crop_refined_source_identity_from_refined_manifest",
    "crop_row_signature_manifest",
    "expected_crop_row_signature_manifest",
    "normalize_crop_metadata_declarations",
    "validate_crop_publication",
    "validate_crop_run_manifest",
]
