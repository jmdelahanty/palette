"""Resolve eye-geometry arrays from canonical subject-mask sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Optional

import numpy as np
import zarr

from .mask_store import MaskStore, MaskStoreError, open_mask_store
from .coordinate_descriptor import (
    COORDINATE_DESCRIPTOR_ATTR,
    parse_canonical_coordinate_descriptor,
)
from .coordinate_frame_record import array_payload_sha256
from .coordinate_identity import ROW_IDENTITY_CONTRACT_ATTR
from .coordinate_record import coordinate_record_sha256
from .coordinate_reference import canonical_node_path
from .json_safety import json_attr_safe
from .provenance_attrs import resolve_source_keypoints_run
from .refined_subject_mask_coordinate_publication import (
    RefinedSubjectMaskCoordinatePublicationError,
    load_persisted_refined_subject_mask_coordinate_surfaces,
)
from .refined_subject_eye_geometry import EYE_COMPONENTS
from .subject_shape_coordinate_publication import (
    SUBJECT_SHAPE_MANIFEST_ATTR,
    SUBJECT_SHAPE_SCALAR_SURFACE_ATTR,
    BoundSubjectShapeCoordinatePublication,
    SubjectShapeCoordinatePublicationError,
    load_persisted_subject_shape_coordinate_publication,
)
from .zarr_run_completion import resolve_authoritative_run_name


EYE_GEOMETRY_STAGE_REFINED_SUBJECT = "refined_subject_masks_runs"
EYE_GEOMETRY_STAGE_SUBJECT_SHAPE = "analysis/subject_shape_runs"
EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_ID = (
    "palette.eye_geometry_staged_subject_shape_authority"
)
EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_VERSION = 1
EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCOPE = (
    "eye_geometry_exact_digest_bound_staged_subset_only"
)

_SUBJECT_SHAPE_EYE_ARRAY_PATHS = (
    "components/eye_left/ellipse_params",
    "components/eye_left/ellipse_success",
    "components/eye_right/ellipse_params",
    "components/eye_right/ellipse_success",
    "relations/eye_pair/separation_px",
)
_SUBJECT_SHAPE_ELLIPSE_PATHS = (
    "components/eye_left/ellipse_params",
    "components/eye_right/ellipse_params",
)
_SUBJECT_SHAPE_EYE_SEPARATION_PATH = "relations/eye_pair/separation_px"
_SUBJECT_SHAPE_SOURCE_CONTRACT_ATTR_NAMES = (
    "schema_id",
    "schema_version",
    "method",
    "method_version",
    "palette_run_completion_status",
    "stage_selector_eligible",
    "coordinate_contract",
    "coordinate_binding_status",
    "subject_shape_publication_owner_uuid",
    "publication_manifest_sha256",
    "source_refined_subject_masks_run",
    "component_names",
    "relation_names",
    "body_frame_schema_id",
    "tail_geometry_schema_id",
)
_ARRAY_MANIFEST_ENTRY_FIELDS = frozenset(
    {
        "array_ref",
        "relative_ref",
        "dtype",
        "shape",
        "content_sha256",
        "canonicalization",
    }
)


def _normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _group_get(group: Any, path: str) -> Any:
    try:
        value = group.get(path)
    except Exception:
        value = None
    if value is not None:
        return value
    try:
        return group[path]
    except Exception:
        return None


def _contains(group: Any, path: str) -> bool:
    return _group_get(group, path) is not None


def _group_names(parent: Any) -> list[str]:
    if parent is None:
        return []
    try:
        names = parent.group_keys()
    except Exception:
        try:
            names = parent.keys()
        except Exception:
            return []
    return sorted(str(name) for name in names)


def _resolve_run_group(
    root: zarr.Group,
    parent_name: str,
    run_name: Optional[str],
    *,
    fallback_to_latest: bool,
) -> tuple[Optional[str], Optional[zarr.Group]]:
    parent = _group_get(root, parent_name)
    if parent is None:
        return None, None

    requested = _normalize_text(run_name)
    if requested == "latest":
        requested = None
    if requested:
        group = _group_get(parent, requested)
        return (requested, group) if group is not None else (requested, None)

    if fallback_to_latest:
        latest = _normalize_text(resolve_authoritative_run_name(parent))
        if latest:
            group = _group_get(parent, latest)
            if group is not None:
                return latest, group

    names = _group_names(parent)
    if names:
        name = names[-1]
        return name, _group_get(parent, name)
    return None, None


def _label_index_map(group: zarr.Group) -> dict[str, int]:
    labels_raw = group.attrs.get("mask_labels")
    if not isinstance(labels_raw, (list, tuple)):
        return {}
    return {str(label): idx for idx, label in enumerate(labels_raw)}


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in getattr(value, "shape", ()))


def _canonical_json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            json_attr_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        json_attr_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _subject_shape_source_contract_attrs(group: Any) -> dict[str, Any]:
    return _canonical_json_copy(
        {
            name: group.attrs.get(name)
            for name in _SUBJECT_SHAPE_SOURCE_CONTRACT_ATTR_NAMES
        }
    )


def _iter_descendant_array_paths(group: Any, prefix: str = ""):
    try:
        array_names = sorted(str(value) for value in group.array_keys())
        group_names = sorted(str(value) for value in group.group_keys())
    except Exception as exc:
        raise ValueError(
            "Staged subject-shape source inventory could not be enumerated."
        ) from exc
    for name in array_names:
        yield f"{prefix}/{name}" if prefix else name
    for name in group_names:
        child = _group_get(group, name)
        if child is None:
            raise ValueError(
                f"Staged subject-shape source group {name!r} disappeared during inventory."
            )
        child_prefix = f"{prefix}/{name}" if prefix else name
        yield from _iter_descendant_array_paths(child, child_prefix)


def _publication_supports_eye_authority(publication: Any) -> bool:
    """Recognize real proof objects while tolerating existing synthetic seams."""

    return all(
        hasattr(publication, name)
        for name in ("manifest", "row_identity", "descriptors", "require_scalar_surface")
    )


def _build_staged_subject_shape_authority(
    group: Any,
    *,
    run_name: str,
    publication: BoundSubjectShapeCoordinatePublication,
) -> dict[str, Any]:
    """Build a detached receipt from one already verified canonical publication."""

    run_path = f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{run_name}"
    if publication.run_path != run_path:
        raise ValueError(
            "Canonical subject-shape publication proof names a different source run."
        )
    row_count = int(publication.row_identity.leading_dimension)
    manifest_arrays = publication.manifest.record.get("arrays")
    if not isinstance(manifest_arrays, Mapping):
        raise ValueError(
            "Canonical subject-shape publication lacks a closed array manifest."
        )

    allowed_arrays: dict[str, Any] = {}
    for relative_ref in _SUBJECT_SHAPE_EYE_ARRAY_PATHS:
        node = _group_get(group, relative_ref)
        manifest_entry = manifest_arrays.get(relative_ref)
        if node is None or not isinstance(manifest_entry, Mapping):
            raise ValueError(
                "Canonical subject-shape publication does not bind required eye "
                f"array {relative_ref!r}."
            )
        expected_array_ref = f"/{run_path}/{relative_ref}"
        if (
            set(manifest_entry) != _ARRAY_MANIFEST_ENTRY_FIELDS
            or manifest_entry.get("relative_ref") != relative_ref
            or manifest_entry.get("array_ref") != expected_array_ref
            or manifest_entry.get("canonicalization")
            != "numpy_dtype_shape_c_order_bytes_v1"
        ):
            raise ValueError(
                f"Canonical manifest entry for eye array {relative_ref!r} is inconsistent."
            )
        allowed_arrays[relative_ref] = _canonical_json_copy(manifest_entry)

    ellipse_descriptors: dict[str, Any] = {}
    for relative_ref in _SUBJECT_SHAPE_ELLIPSE_PATHS:
        binding = publication.descriptors.get(relative_ref)
        if binding is None:
            raise ValueError(
                "Canonical subject-shape publication lacks ellipse coordinate "
                f"descriptor {relative_ref!r}."
            )
        expected_path = f"{run_path}/{relative_ref}"
        if canonical_node_path(binding.coordinate_node) != expected_path:
            raise ValueError(
                f"Ellipse coordinate descriptor {relative_ref!r} binds another array."
            )
        ellipse_descriptors[relative_ref] = {
            "record_ref": f"/{expected_path}@{COORDINATE_DESCRIPTOR_ATTR}",
            "descriptor_sha256": binding.descriptor.digest(),
        }

    separation = publication.require_scalar_surface(
        _SUBJECT_SHAPE_EYE_SEPARATION_PATH,
        units="px",
        surface_kind="row_scalar",
    )
    if canonical_node_path(separation.array_node) != (
        f"{run_path}/{_SUBJECT_SHAPE_EYE_SEPARATION_PATH}"
    ):
        raise ValueError(
            "Canonical eye-separation scalar semantics bind another array."
        )

    publication_source = getattr(publication, "source", None)
    source_context = getattr(publication_source, "context", None)
    assignment_authority = getattr(
        source_context,
        "assignment_keypoint_authority",
        None,
    )
    assignment_pointer = None
    if assignment_authority is not None:
        assignment_ref = getattr(assignment_authority, "record_ref", None)
        assignment_sha256 = getattr(assignment_authority, "record_sha256", None)
        if (
            not isinstance(assignment_ref, str)
            or not assignment_ref.startswith("/")
            or not _is_sha256(assignment_sha256)
        ):
            raise ValueError(
                "Canonical subject-shape assignment keypoint authority is malformed."
            )
        assignment_pointer = {
            "record_ref": assignment_ref,
            "record_sha256": assignment_sha256,
        }
    else:
        rebinding = getattr(
            publication_source,
            "assignment_keypoint_rebinding_manifest",
            None,
        )
        rebinding_run_id = getattr(
            publication_source,
            "assignment_keypoint_rebinding_run_id",
            None,
        )
        if isinstance(rebinding, Mapping) and isinstance(rebinding_run_id, str):
            assignment_pointer = {
                "record_ref": (
                    "/subject_mask_assignment_keypoint_rebinding_runs/"
                    f"{rebinding_run_id}@run_manifest"
                ),
                "record_sha256": _canonical_sha256(rebinding),
            }

    record = {
        "schema_id": EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_ID,
        "schema_version": EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCOPE,
        "source_subject_shape_run": run_name,
        "source_subject_shape_run_ref": f"/{run_path}",
        "row_count": row_count,
        "canonical_publication": {
            "manifest_ref": publication.manifest.record_ref,
            "manifest_sha256": publication.manifest.record_sha256,
            "row_identity_ref": publication.row_identity.record_ref,
            "row_identity_sha256": publication.row_identity.record_sha256,
            "ellipse_coordinate_descriptors": ellipse_descriptors,
            "separation_scalar_surface": {
                "record_ref": separation.semantics.record_ref,
                "record_sha256": separation.semantics.record_sha256,
            },
            "assignment_keypoint_authority": assignment_pointer,
        },
        "source_contract_attrs": _subject_shape_source_contract_attrs(group),
        "allowed_arrays": allowed_arrays,
        "closed_array_inventory": True,
        "normal_reader_authority": False,
    }
    return {
        **record,
        "record_sha256": _canonical_sha256(record),
    }


def _validate_descriptor_metadata(
    node: Any,
    *,
    relative_ref: str,
    proof: Mapping[str, Any],
    row_identity_ref: str,
    row_identity_sha256: str,
) -> None:
    expected_digest = proof.get("descriptor_sha256")
    raw = node.attrs.get(COORDINATE_DESCRIPTOR_ATTR)
    try:
        descriptor = parse_canonical_coordinate_descriptor(raw)
        exact_descriptor = _canonical_json_copy(raw)
    except Exception as exc:
        raise ValueError(
            f"Staged ellipse descriptor metadata for {relative_ref!r} is invalid: {exc}"
        ) from exc
    if (
        exact_descriptor != descriptor.to_dict()
        or descriptor.digest() != expected_digest
        or node.attrs.get(f"{COORDINATE_DESCRIPTOR_ATTR}_sha256")
        != expected_digest
        or node.attrs.get(f"{COORDINATE_DESCRIPTOR_ATTR}_owner_dtype")
        != np.dtype(node.dtype).str
        or descriptor.row_identity.record_ref != row_identity_ref
        or descriptor.row_identity.record_sha256 != row_identity_sha256
    ):
        raise ValueError(
            f"Staged ellipse descriptor metadata for {relative_ref!r} differs from its receipt."
        )


def _validate_scalar_surface_metadata(
    node: Any,
    *,
    proof: Mapping[str, Any],
    row_identity_ref: str,
    row_identity_sha256: str,
) -> None:
    expected_digest = proof.get("record_sha256")
    raw = node.attrs.get(SUBJECT_SHAPE_SCALAR_SURFACE_ATTR)
    try:
        observed_digest = coordinate_record_sha256(raw)
    except Exception as exc:
        raise ValueError(
            "Staged eye-separation scalar-surface metadata is invalid."
        ) from exc
    if (
        observed_digest != expected_digest
        or node.attrs.get(f"{SUBJECT_SHAPE_SCALAR_SURFACE_ATTR}_sha256")
        != expected_digest
        or not isinstance(raw, Mapping)
        or raw.get("row_identity")
        != {
            "record_ref": row_identity_ref,
            "record_sha256": row_identity_sha256,
        }
    ):
        raise ValueError(
            "Staged eye-separation scalar-surface metadata differs from its receipt."
        )


def _validated_staged_subject_shape_authority(
    group: Any,
    *,
    run_name: str,
    authority: Mapping[str, Any],
    verify_payload: bool,
) -> dict[str, Any]:
    """Validate the materializer-only authority for an exact staged subset."""

    if not isinstance(authority, Mapping):
        raise ValueError("Staged subject-shape authority must be a mapping.")
    if type(verify_payload) is not bool:
        raise ValueError("Staged payload verification flag must be an exact bool.")
    canonical = _canonical_json_copy(authority)
    digest = canonical.pop("record_sha256", None)
    expected_fields = {
        "schema_id",
        "schema_version",
        "authority_scope",
        "source_subject_shape_run",
        "source_subject_shape_run_ref",
        "row_count",
        "canonical_publication",
        "source_contract_attrs",
        "allowed_arrays",
        "closed_array_inventory",
        "normal_reader_authority",
    }
    if set(canonical) != expected_fields:
        raise ValueError("Staged subject-shape authority fields are not exact.")
    if not _is_sha256(digest) or digest != _canonical_sha256(canonical):
        raise ValueError("Staged subject-shape authority digest is missing or stale.")
    if canonical.get("schema_id") != EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_ID:
        raise ValueError("Unsupported staged subject-shape authority schema.")
    if (
        type(canonical.get("schema_version")) is not int
        or canonical.get("schema_version")
        != EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported staged subject-shape authority schema version.")
    if canonical.get("authority_scope") != EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCOPE:
        raise ValueError("Staged subject-shape authority has the wrong scope.")
    if canonical.get("normal_reader_authority") is not False:
        raise ValueError("Detached staging receipts cannot grant normal reader authority.")
    if canonical.get("closed_array_inventory") is not True:
        raise ValueError("Staged subject-shape array inventory is not closed.")

    run_path = f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{run_name}"
    if (
        canonical.get("source_subject_shape_run") != run_name
        or canonical.get("source_subject_shape_run_ref") != f"/{run_path}"
    ):
        raise ValueError("Staged subject-shape authority names a different source run.")
    row_count = canonical.get("row_count")
    if type(row_count) is not int or row_count < 0:
        raise ValueError("Staged subject-shape authority has an invalid row count.")

    publication = canonical.get("canonical_publication")
    publication_fields = {
        "manifest_ref",
        "manifest_sha256",
        "row_identity_ref",
        "row_identity_sha256",
        "ellipse_coordinate_descriptors",
        "separation_scalar_surface",
        "assignment_keypoint_authority",
    }
    if not isinstance(publication, Mapping) or set(publication) != publication_fields:
        raise ValueError(
            "Staged authority lacks exact canonical-publication proof fields."
        )
    if (
        publication.get("manifest_ref")
        != f"/{run_path}@{SUBJECT_SHAPE_MANIFEST_ATTR}"
        or publication.get("row_identity_ref")
        != f"/{run_path}@{ROW_IDENTITY_CONTRACT_ATTR}"
        or not _is_sha256(publication.get("manifest_sha256"))
        or not _is_sha256(publication.get("row_identity_sha256"))
    ):
        raise ValueError(
            "Staged authority canonical manifest or row-identity proof is invalid."
        )

    attrs = canonical.get("source_contract_attrs")
    if (
        not isinstance(attrs, Mapping)
        or set(attrs) != set(_SUBJECT_SHAPE_SOURCE_CONTRACT_ATTR_NAMES)
        or attrs != _subject_shape_source_contract_attrs(group)
        or attrs.get("palette_run_completion_status") != "complete"
        or attrs.get("stage_selector_eligible") is not True
        or attrs.get("coordinate_contract") != "canonical_v2"
        or attrs.get("coordinate_binding_status") != "bound_canonical_v2"
        or attrs.get("publication_manifest_sha256")
        != publication.get("manifest_sha256")
        or not _normalize_text(attrs.get("subject_shape_publication_owner_uuid"))
    ):
        raise ValueError(
            "Staged source contract attrs differ from the canonical receipt."
        )

    ellipse_proofs = publication.get("ellipse_coordinate_descriptors")
    if (
        not isinstance(ellipse_proofs, Mapping)
        or set(ellipse_proofs) != set(_SUBJECT_SHAPE_ELLIPSE_PATHS)
    ):
        raise ValueError("Staged authority lacks exact ellipse descriptor proofs.")
    for relative_ref in _SUBJECT_SHAPE_ELLIPSE_PATHS:
        proof = ellipse_proofs.get(relative_ref)
        if (
            not isinstance(proof, Mapping)
            or set(proof) != {"record_ref", "descriptor_sha256"}
            or proof.get("record_ref")
            != f"/{run_path}/{relative_ref}@{COORDINATE_DESCRIPTOR_ATTR}"
            or not _is_sha256(proof.get("descriptor_sha256"))
        ):
            raise ValueError(
                f"Staged ellipse descriptor proof for {relative_ref!r} is invalid."
            )

    separation_proof = publication.get("separation_scalar_surface")
    if (
        not isinstance(separation_proof, Mapping)
        or set(separation_proof) != {"record_ref", "record_sha256"}
        or separation_proof.get("record_ref")
        != (
            f"/{run_path}/{_SUBJECT_SHAPE_EYE_SEPARATION_PATH}"
            f"@{SUBJECT_SHAPE_SCALAR_SURFACE_ATTR}"
        )
        or not _is_sha256(separation_proof.get("record_sha256"))
    ):
        raise ValueError("Staged eye-separation scalar-surface proof is invalid.")

    assignment_proof = publication.get("assignment_keypoint_authority")
    if assignment_proof is not None and (
        not isinstance(assignment_proof, Mapping)
        or set(assignment_proof) != {"record_ref", "record_sha256"}
        or not isinstance(assignment_proof.get("record_ref"), str)
        or not assignment_proof["record_ref"].startswith("/")
        or not _is_sha256(assignment_proof.get("record_sha256"))
    ):
        raise ValueError(
            "Staged subject-shape assignment-keypoint authority pointer is invalid."
        )

    allowed = canonical.get("allowed_arrays")
    if not isinstance(allowed, Mapping) or set(allowed) != set(
        _SUBJECT_SHAPE_EYE_ARRAY_PATHS
    ):
        raise ValueError(
            "Staged subject-shape authority has missing or unsupported arrays."
        )
    observed_inventory = set(_iter_descendant_array_paths(group))
    if observed_inventory != set(_SUBJECT_SHAPE_EYE_ARRAY_PATHS):
        missing = sorted(set(_SUBJECT_SHAPE_EYE_ARRAY_PATHS) - observed_inventory)
        extra = sorted(observed_inventory - set(_SUBJECT_SHAPE_EYE_ARRAY_PATHS))
        raise ValueError(
            "Staged subject-shape subset has a noncanonical array inventory: "
            f"missing={missing!r}, extra={extra!r}."
        )

    for relative_ref in _SUBJECT_SHAPE_EYE_ARRAY_PATHS:
        declared = allowed.get(relative_ref)
        node = _group_get(group, relative_ref)
        if not isinstance(declared, Mapping) or node is None:
            raise ValueError(
                f"Staged source array {relative_ref!r} is missing from receipt or subset."
            )
        expected_shape = declared.get("shape")
        if (
            set(declared) != _ARRAY_MANIFEST_ENTRY_FIELDS
            or declared.get("array_ref") != f"/{run_path}/{relative_ref}"
            or declared.get("relative_ref") != relative_ref
            or declared.get("canonicalization")
            != "numpy_dtype_shape_c_order_bytes_v1"
            or not isinstance(expected_shape, list)
            or not expected_shape
            or any(type(value) is not int or value < 0 for value in expected_shape)
            or expected_shape[0] != row_count
            or declared.get("dtype") != np.dtype(node.dtype).str
            or expected_shape != [int(value) for value in node.shape]
            or not _is_sha256(declared.get("content_sha256"))
        ):
            raise ValueError(
                f"Staged source metadata for {relative_ref!r} differs from its receipt."
            )
        if relative_ref in _SUBJECT_SHAPE_ELLIPSE_PATHS:
            _validate_descriptor_metadata(
                node,
                relative_ref=relative_ref,
                proof=ellipse_proofs[relative_ref],
                row_identity_ref=str(publication["row_identity_ref"]),
                row_identity_sha256=str(publication["row_identity_sha256"]),
            )
        elif relative_ref == _SUBJECT_SHAPE_EYE_SEPARATION_PATH:
            _validate_scalar_surface_metadata(
                node,
                proof=separation_proof,
                row_identity_ref=str(publication["row_identity_ref"]),
                row_identity_sha256=str(publication["row_identity_sha256"]),
            )
        if verify_payload:
            try:
                observed_digest = array_payload_sha256(node)
            except Exception as exc:
                raise ValueError(
                    f"Staged source array {relative_ref!r} could not be verified: {exc}"
                ) from exc
            if observed_digest != declared.get("content_sha256"):
                raise ValueError(
                    f"Staged source array {relative_ref!r} differs from its canonical payload."
                )

    return {**canonical, "record_sha256": str(digest)}


def _select_channel_indices(channel_key: object, channel_count: int) -> list[int]:
    if isinstance(channel_key, (int, np.integer)):
        idx = int(channel_key)
        if idx < 0:
            idx += channel_count
        return [idx]
    if isinstance(channel_key, slice):
        return list(range(channel_count))[channel_key]
    if isinstance(channel_key, np.ndarray):
        return [int(v) for v in channel_key.reshape(-1).tolist()]
    if isinstance(channel_key, Sequence) and not isinstance(channel_key, (str, bytes, bytearray)):
        return [int(v) for v in channel_key]
    raise TypeError(f"Unsupported channel index type: {type(channel_key).__name__}")


def _is_scalar_row_key(row_key: object) -> bool:
    return isinstance(row_key, (int, np.integer))


def _stack_component_values(values: list[np.ndarray], *, row_scalar: bool) -> np.ndarray:
    if not values:
        return np.asarray(values)
    axis = 0 if row_scalar else 1
    return np.stack(values, axis=axis)


class StackedComponentArray:
    """Array-like view that exposes split component arrays as ``(N, C, ...)``."""

    def __init__(self, components: Sequence[Any]):
        if not components:
            raise ValueError("At least one component array is required.")
        shapes = [_shape(component) for component in components]
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError(f"Component shapes do not match: {shapes}")
        self._components = tuple(components)
        self._component_shape = shapes[0]
        self.shape = (
            int(self._component_shape[0]),
            len(self._components),
            *tuple(int(v) for v in self._component_shape[1:]),
        )
        self.ndim = len(self.shape)
        self.dtype = getattr(self._components[0], "dtype", np.asarray(self._components[0][:]).dtype)
        chunks = getattr(self._components[0], "chunks", None)
        self.chunks = None if chunks is None else (chunks[0], len(self._components), *tuple(chunks[1:]))

    def __array__(self, dtype=None) -> np.ndarray:
        values = np.asarray(self[:])
        return values.astype(dtype, copy=False) if dtype is not None else values

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            values = [np.asarray(component[key]) for component in self._components]
            return _stack_component_values(values, row_scalar=_is_scalar_row_key(key))

        if not key:
            return self[:]
        row_key = key[0]
        if len(key) == 1:
            values = [np.asarray(component[row_key]) for component in self._components]
            return _stack_component_values(values, row_scalar=_is_scalar_row_key(row_key))

        channel_key = key[1]
        tail_key = tuple(key[2:])
        channel_indices = _select_channel_indices(channel_key, len(self._components))
        component_key = (row_key, *tail_key)
        values = [np.asarray(self._components[idx][component_key]) for idx in channel_indices]
        if isinstance(channel_key, (int, np.integer)):
            return np.asarray(values[0])
        return _stack_component_values(values, row_scalar=_is_scalar_row_key(row_key))


class ChannelSelectionArray:
    """Array-like view selecting semantic component channels from ``masks_roi``."""

    def __init__(self, source: Any, channel_indices: Sequence[int]):
        if not channel_indices:
            raise ValueError("At least one channel index is required.")
        self._source = source
        self._channel_indices = tuple(int(v) for v in channel_indices)
        source_shape = _shape(source)
        if len(source_shape) < 2:
            raise ValueError(f"Source array must include a channel axis, got {source_shape}.")
        self._component_shape = (int(source_shape[0]), *tuple(int(v) for v in source_shape[2:]))
        self.shape = (int(source_shape[0]), len(self._channel_indices), *tuple(int(v) for v in source_shape[2:]))
        self.ndim = len(self.shape)
        self.dtype = getattr(source, "dtype", np.asarray(source[:1]).dtype)
        chunks = getattr(source, "chunks", None)
        self.chunks = None if chunks is None else (chunks[0], len(self._channel_indices), *tuple(chunks[2:]))

    def __array__(self, dtype=None) -> np.ndarray:
        values = np.asarray(self[:])
        return values.astype(dtype, copy=False) if dtype is not None else values

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            values = [np.asarray(self._source[(key, idx)]) for idx in self._channel_indices]
            return _stack_component_values(values, row_scalar=_is_scalar_row_key(key))

        if not key:
            return self[:]
        row_key = key[0]
        if len(key) == 1:
            values = [np.asarray(self._source[(row_key, idx)]) for idx in self._channel_indices]
            return _stack_component_values(values, row_scalar=_is_scalar_row_key(row_key))

        channel_key = key[1]
        tail_key = tuple(key[2:])
        selected = _select_channel_indices(channel_key, len(self._channel_indices))
        source_key_tail = (row_key, *tail_key)
        values = [
            np.asarray(self._source[(source_key_tail[0], self._channel_indices[idx], *source_key_tail[1:])])
            for idx in selected
        ]
        if isinstance(channel_key, (int, np.integer)):
            return np.asarray(values[0])
        return _stack_component_values(values, row_scalar=_is_scalar_row_key(row_key))


class MaskStoreChannelSelectionArray:
    """Array-like view selecting semantic component channels from ``MaskStore``."""

    def __init__(self, mask_store: MaskStore, channel_indices: Sequence[int]):
        if not channel_indices:
            raise ValueError("At least one channel index is required.")
        self._mask_store = mask_store
        self._channel_indices = tuple(int(v) for v in channel_indices)
        if any(idx < 0 or idx >= int(mask_store.shape[1]) for idx in self._channel_indices):
            raise ValueError(f"Channel indices {self._channel_indices} out of range for mask store {mask_store.shape}.")
        self.shape = (
            int(mask_store.shape[0]),
            len(self._channel_indices),
            int(mask_store.shape[2]),
            int(mask_store.shape[3]),
        )
        self.ndim = len(self.shape)
        self.dtype = np.dtype(np.uint8)
        self.chunks = None

    def __array__(self, dtype=None) -> np.ndarray:
        values = np.asarray(self[:])
        return values.astype(dtype, copy=False) if dtype is not None else values

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            values = self._mask_store.read_dense(rows=key, channels=self._channel_indices)
            return values[0] if _is_scalar_row_key(key) else values

        if not key:
            return self[:]
        row_key = key[0]
        if len(key) == 1:
            values = self._mask_store.read_dense(rows=row_key, channels=self._channel_indices)
            return values[0] if _is_scalar_row_key(row_key) else values

        channel_key = key[1]
        tail_key = tuple(key[2:])
        selected = _select_channel_indices(channel_key, len(self._channel_indices))
        source_channels = [self._channel_indices[idx] for idx in selected]
        values = self._mask_store.read_dense(rows=row_key, channels=source_channels)

        row_scalar = _is_scalar_row_key(row_key)
        channel_scalar = isinstance(channel_key, (int, np.integer))
        if row_scalar:
            values = values[0]
        if channel_scalar:
            values = values[0] if row_scalar else values[:, 0]
        if tail_key:
            if row_scalar and channel_scalar:
                values = values[tail_key]
            elif row_scalar or channel_scalar:
                values = values[(slice(None), *tail_key)]
            else:
                values = values[(slice(None), slice(None), *tail_key)]
        return values


@dataclass
class EyeGeometrySource:
    stage_group: str
    run_name: str
    group_path: str
    group: zarr.Group
    masks_roi: Optional[Any]
    ellipse_params: Any
    ellipse_success: Any
    eye_separation: Optional[Any]
    lineage_attrs: Mapping[str, object]
    source_refined_eye_run: Optional[str] = None
    source_refined_subject_run: Optional[str] = None
    source_subject_shape_run: Optional[str] = None
    coordinate_authority_status: str = "canonical"
    source_authority_mode: Optional[str] = None
    source_authority: Optional[Mapping[str, Any]] = field(default=None, repr=False)
    subject_shape_coordinate_publication: Optional[
        BoundSubjectShapeCoordinatePublication
    ] = field(default=None, repr=False, compare=False)


def _has_subject_eye_geometry(group: zarr.Group) -> bool:
    label_map = _label_index_map(group)
    if any(component not in label_map for component in EYE_COMPONENTS):
        return False
    try:
        mask_store = open_mask_store(group, prefer="dense")
    except (MaskStoreError, ValueError):
        return False
    if len(mask_store.shape) != 4:
        return False
    channel_count = int(mask_store.shape[1])
    if any(int(label_map[component]) >= channel_count for component in EYE_COMPONENTS):
        return False
    for component in EYE_COMPONENTS:
        if not _contains(group, f"components/{component}/geometry/ellipse_params"):
            return False
        if not _contains(group, f"components/{component}/geometry/ellipse_success"):
            return False
    return _contains(group, "relations/eye_pair/metrics/separation_px")


def _has_subject_shape_eye_geometry(group: zarr.Group) -> bool:
    arrays: list[Any] = []
    for component in EYE_COMPONENTS:
        params = _group_get(group, f"components/{component}/ellipse_params")
        success = _group_get(group, f"components/{component}/ellipse_success")
        if params is None or success is None:
            return False
        param_shape = _shape(params)
        success_shape = _shape(success)
        if len(param_shape) < 2 or len(success_shape) != 1 or param_shape[0] != success_shape[0]:
            return False
        arrays.extend([params, success])
    if _shape(arrays[0]) != _shape(arrays[2]) or _shape(arrays[1]) != _shape(arrays[3]):
        return False
    separation = _group_get(group, "relations/eye_pair/separation_px")
    return separation is not None and _shape(separation)[:1] == _shape(arrays[1])[:1]


def _find_latest_subject_eye_geometry(root: zarr.Group) -> tuple[Optional[str], Optional[zarr.Group]]:
    parent = _group_get(root, EYE_GEOMETRY_STAGE_REFINED_SUBJECT)
    if parent is None:
        return None, None

    latest = _normalize_text(resolve_authoritative_run_name(parent))
    if latest:
        latest_group = _group_get(parent, latest)
        if latest_group is not None and _has_subject_eye_geometry(latest_group):
            return latest, latest_group

    for name in reversed(_group_names(parent)):
        group = _group_get(parent, name)
        if group is not None and _has_subject_eye_geometry(group):
            return name, group
    return None, None


def _find_latest_subject_shape_eye_geometry(
    root: zarr.Group,
) -> tuple[
    Optional[str],
    Optional[zarr.Group],
    Optional[BoundSubjectShapeCoordinatePublication],
]:
    parent = _group_get(root, EYE_GEOMETRY_STAGE_SUBJECT_SHAPE)
    if parent is None:
        return None, None, None
    latest = _normalize_text(parent.attrs.get("latest"))
    if latest is None:
        raise ValueError(
            "analysis/subject_shape_runs has no exact latest publication selector."
        )
    return _strict_subject_shape_eye_geometry(root, latest)


def _normalized_run_name(value: Optional[str]) -> Optional[str]:
    text = _normalize_text(value)
    if text is None or text.lower() == "latest":
        return None
    return text.strip("/").rsplit("/", 1)[-1]


def _exact_staged_subject_shape_run_name(value: Optional[str]) -> str:
    requested = _normalize_text(value)
    if requested is None or requested.lower() == "latest":
        raise ValueError(
            "Digest-bound staged eye geometry requires an exact subject-shape run name."
        )
    normalized = requested.strip("/")
    prefix = f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/"
    run_name = normalized[len(prefix) :] if normalized.startswith(prefix) else normalized
    if not run_name or "/" in run_name:
        raise ValueError(f"Invalid staged subject-shape run name {value!r}.")
    return run_name


def _strict_subject_shape_eye_geometry(
    root: zarr.Group,
    run_name: Optional[str],
) -> tuple[str, zarr.Group, BoundSubjectShapeCoordinatePublication]:
    parent = _group_get(root, EYE_GEOMETRY_STAGE_SUBJECT_SHAPE)
    if parent is None:
        raise ValueError("Archive has no analysis/subject_shape_runs group.")
    selected = _normalized_run_name(run_name) or _normalize_text(
        parent.attrs.get("latest")
    )
    if selected is None:
        raise ValueError(
            "analysis/subject_shape_runs has no exact latest publication selector."
        )
    path = f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{selected}"
    try:
        publication = load_persisted_subject_shape_coordinate_publication(
            root,
            path,
        )
    except (SubjectShapeCoordinatePublicationError, KeyError, ValueError) as exc:
        raise ValueError(
            f"Subject-shape eye source {path!r} is not a canonical publication: {exc}"
        ) from exc
    group = _group_get(root, path)
    if group is None or not _has_subject_shape_eye_geometry(group):
        raise ValueError(f"{path} is missing canonical subject-shape eye geometry.")
    return selected, group, publication


def _strict_refined_subject_eye_geometry(
    root: zarr.Group,
    run_name: Optional[str],
) -> tuple[str, zarr.Group]:
    parent = _group_get(root, EYE_GEOMETRY_STAGE_REFINED_SUBJECT)
    if parent is None:
        raise ValueError("Archive has no refined_subject_masks_runs group.")
    selected = _normalized_run_name(run_name) or _normalize_text(
        parent.attrs.get("latest")
    )
    if selected is None:
        raise ValueError(
            "refined_subject_masks_runs has no exact latest publication selector."
        )
    path = f"{EYE_GEOMETRY_STAGE_REFINED_SUBJECT}/{selected}"
    try:
        load_persisted_refined_subject_mask_coordinate_surfaces(root, path)
    except (RefinedSubjectMaskCoordinatePublicationError, KeyError, ValueError) as exc:
        raise ValueError(
            f"Refined-subject eye source {path!r} is not a canonical publication: {exc}"
        ) from exc
    group = _group_get(root, path)
    if group is None or not _has_subject_eye_geometry(group):
        raise ValueError(f"{path} is missing canonical refined-subject eye geometry.")
    return selected, group


def _build_subject_source(
    run_name: str,
    group: zarr.Group,
    *,
    source_refined_eye_run: Optional[str] = None,
    historical_compatibility: bool = False,
) -> EyeGeometrySource:
    if not _has_subject_eye_geometry(group):
        raise ValueError(f"{EYE_GEOMETRY_STAGE_REFINED_SUBJECT}/{run_name} missing canonical eye geometry.")
    label_map = _label_index_map(group)
    mask_store = open_mask_store(
        group,
        source_path=f"{EYE_GEOMETRY_STAGE_REFINED_SUBJECT}/{run_name}",
        prefer="dense",
    )
    masks = MaskStoreChannelSelectionArray(mask_store, [label_map["eye_left"], label_map["eye_right"]])
    ellipse_params = StackedComponentArray(
        [
            group["components/eye_left/geometry/ellipse_params"],
            group["components/eye_right/geometry/ellipse_params"],
        ]
    )
    ellipse_success = StackedComponentArray(
        [
            group["components/eye_left/geometry/ellipse_success"],
            group["components/eye_right/geometry/ellipse_success"],
        ]
    )
    source_refined_eye_run = source_refined_eye_run or _normalize_text(group.attrs.get("source_refined_eye_masks_run"))
    lineage_attrs = dict(group.attrs)
    if historical_compatibility:
        lineage_attrs["coordinate_authority_status"] = (
            "historical_compatibility_noncanonical"
        )
    return EyeGeometrySource(
        stage_group=EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
        run_name=run_name,
        group_path=f"{EYE_GEOMETRY_STAGE_REFINED_SUBJECT}/{run_name}",
        group=group,
        masks_roi=masks,
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        eye_separation=group["relations/eye_pair/metrics/separation_px"],
        lineage_attrs=lineage_attrs,
        source_refined_eye_run=source_refined_eye_run,
        source_refined_subject_run=run_name,
        coordinate_authority_status=(
            "historical_compatibility_noncanonical"
            if historical_compatibility
            else "canonical"
        ),
    )


def _build_subject_shape_source(
    run_name: str,
    group: zarr.Group,
    *,
    publication: Any = None,
    source_authority: Optional[Mapping[str, Any]] = None,
    source_authority_mode: str = "canonical_publication",
) -> EyeGeometrySource:
    if not _has_subject_shape_eye_geometry(group):
        raise ValueError(f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{run_name} missing analysis eye geometry.")
    ellipse_params = StackedComponentArray(
        [
            group["components/eye_left/ellipse_params"],
            group["components/eye_right/ellipse_params"],
        ]
    )
    ellipse_success = StackedComponentArray(
        [
            group["components/eye_left/ellipse_success"],
            group["components/eye_right/ellipse_success"],
        ]
    )
    if source_authority is None and _publication_supports_eye_authority(
        publication
    ):
        source_authority = _build_staged_subject_shape_authority(
            group,
            run_name=run_name,
            publication=publication,
        )
    return EyeGeometrySource(
        stage_group=EYE_GEOMETRY_STAGE_SUBJECT_SHAPE,
        run_name=run_name,
        group_path=f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{run_name}",
        group=group,
        masks_roi=None,
        ellipse_params=ellipse_params,
        ellipse_success=ellipse_success,
        eye_separation=group["relations/eye_pair/separation_px"],
        lineage_attrs=dict(group.attrs),
        source_refined_eye_run=_normalize_text(group.attrs.get("source_refined_eye_masks_run")),
        source_refined_subject_run=_normalize_text(group.attrs.get("source_refined_subject_masks_run")),
        source_subject_shape_run=run_name,
        source_authority_mode=source_authority_mode,
        source_authority=source_authority,
        subject_shape_coordinate_publication=publication,
    )


def resolve_eye_geometry_source(
    root: zarr.Group,
    *,
    subject_shape_run: Optional[str] = None,
    refined_subject_run: Optional[str] = None,
    prefer_subject_shape: bool = False,
    prefer_subject: bool = True,
    historical_refined_subject_compatibility: bool = False,
    _staged_subject_shape_authority: Optional[Mapping[str, Any]] = None,
    _verify_staged_payload: bool = True,
) -> EyeGeometrySource:
    """Resolve the active eye geometry source.

    Subject-shape and refined-subject normal reads both require strict
    canonical publication reloads. Historical refined-subject geometry is
    available only through the explicitly noncanonical compatibility flag;
    it must not be used as future scientific coordinate authority.

    The private staged path is reserved for materializers. It accepts only an
    exact named subject-shape run plus a closed digest-bound receipt previously
    derived from the fully verified canonical publication. The receipt cannot
    authorize normal readers.
    """

    if _staged_subject_shape_authority is not None:
        if refined_subject_run is not None or historical_refined_subject_compatibility:
            raise ValueError(
                "Digest-bound staged eye geometry cannot resolve a refined-subject source."
            )
        run_name = _exact_staged_subject_shape_run_name(subject_shape_run)
        path = f"{EYE_GEOMETRY_STAGE_SUBJECT_SHAPE}/{run_name}"
        group = _group_get(root, path)
        if group is None:
            raise ValueError(f"Staged subject-shape run {path!r} is missing.")
        authority = _validated_staged_subject_shape_authority(
            group,
            run_name=run_name,
            authority=_staged_subject_shape_authority,
            verify_payload=_verify_staged_payload,
        )
        if not _has_subject_shape_eye_geometry(group):
            raise ValueError(
                f"{path} is missing staged subject-shape eye geometry."
            )
        return _build_subject_shape_source(
            run_name,
            group,
            publication=None,
            source_authority=authority,
            source_authority_mode="digest_bound_staged_subset",
        )

    if _verify_staged_payload is not True:
        raise ValueError(
            "_verify_staged_payload is private to digest-bound staged resolution."
        )

    if subject_shape_run:
        run_name, group, publication = _strict_subject_shape_eye_geometry(
            root,
            subject_shape_run,
        )
        return _build_subject_shape_source(
            run_name,
            group,
            publication=publication,
        )

    if refined_subject_run:
        if historical_refined_subject_compatibility:
            run_name, group = _resolve_run_group(
                root,
                EYE_GEOMETRY_STAGE_REFINED_SUBJECT,
                refined_subject_run,
                fallback_to_latest=False,
            )
            if run_name is None or group is None:
                raise ValueError(
                    f"Historical refined subject-mask run not found: {refined_subject_run}."
                )
        else:
            run_name, group = _strict_refined_subject_eye_geometry(
                root,
                refined_subject_run,
            )
        source_refined_eye_run = _normalize_text(group.attrs.get("source_refined_eye_masks_run"))
        return _build_subject_source(
            run_name,
            group,
            source_refined_eye_run=source_refined_eye_run,
            historical_compatibility=historical_refined_subject_compatibility,
        )

    if prefer_subject_shape:
        shape_name, shape_group, publication = _find_latest_subject_shape_eye_geometry(
            root
        )
        if shape_name is not None and shape_group is not None:
            return _build_subject_shape_source(
                shape_name,
                shape_group,
                publication=publication,
            )

    if prefer_subject:
        if historical_refined_subject_compatibility:
            subject_name, subject_group = _find_latest_subject_eye_geometry(root)
        else:
            try:
                subject_name, subject_group = _strict_refined_subject_eye_geometry(
                    root,
                    None,
                )
            except ValueError:
                subject_name, subject_group = None, None
        if subject_name is not None and subject_group is not None:
            source_refined_eye_run = _normalize_text(subject_group.attrs.get("source_refined_eye_masks_run"))
            return _build_subject_source(
                subject_name,
                subject_group,
                source_refined_eye_run=source_refined_eye_run,
                historical_compatibility=historical_refined_subject_compatibility,
            )

    raise ValueError("No canonical subject-shape or refined-subject eye geometry found.")


def resolve_source_keypoints_run_for_eye_geometry(source: EyeGeometrySource) -> Optional[str]:
    return resolve_source_keypoints_run(source.lineage_attrs)


__all__ = [
    "ChannelSelectionArray",
    "EYE_GEOMETRY_STAGE_REFINED_SUBJECT",
    "EYE_GEOMETRY_STAGE_SUBJECT_SHAPE",
    "EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_ID",
    "EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCHEMA_VERSION",
    "EYE_GEOMETRY_STAGED_SUBJECT_SHAPE_AUTHORITY_SCOPE",
    "EyeGeometrySource",
    "StackedComponentArray",
    "resolve_eye_geometry_source",
    "resolve_source_keypoints_run_for_eye_geometry",
]
