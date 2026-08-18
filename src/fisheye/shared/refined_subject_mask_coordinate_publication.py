"""Canonical coordinate publication for future refined subject-mask runs.

Future-normal refined runs are exact, dense, observation-row-aligned derivatives
of one canonical ``subject_mask_runs`` child.  The raster values remain
ROI-local.  This module binds every authoritative raster and persisted point,
bbox, ellipse, or contour surface to the refined row identity, exact ROI
extent, direction-labelled ROI-to-source-camera placement, selected raw-mask
authority, and exact refinement provenance.

Historical compact-only or implicitly framed runs are intentionally outside
this API.  They remain inspectable through the explicitly permissive logical
reader in :mod:`fisheye.shared.refined_subject_masks_io`, but cannot pass this
publication gate or become a future-normal scientific input.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import re
from threading import RLock
from typing import Any, Mapping, Sequence

import numpy as np

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.array_measurement_descriptor import (
    ARRAY_MEASUREMENT_DESCRIPTOR_ATTR,
    build_array_measurement_descriptor,
    load_bound_array_measurement_descriptor,
    stamp_and_bind_array_measurement_descriptor,
)
from fisheye.shared.canonical_coordinate_publication import (
    BoundCanonicalCoordinateDescriptor,
    build_bound_canonical_coordinate_descriptor,
    load_bound_canonical_coordinate_descriptor,
    stamp_bound_canonical_coordinate_descriptors,
)
from fisheye.shared.coordinate_descriptor import (
    CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
    CanonicalCollectionAxis,
    DigestBoundCoordinateRecordRef,
)
from fisheye.shared.keypoint_coordinate_publication import (
    BoundKeypointCoordinateSurfaces,
    load_persisted_keypoint_coordinate_surfaces,
    require_bound_keypoint_coordinate_surfaces,
)
from fisheye.shared.coordinate_identity import (
    BoundRowIdentityContract,
    BoundSourceRowTemporalAuthority,
    OBSERVATION_INSTANCE_DOMAIN,
    build_row_identity_contract,
    load_bound_row_identity_contract,
    load_bound_source_row_temporal_authority,
    stamp_and_bind_row_identity_contract,
    stamp_source_row_temporal_authority,
)
from fisheye.shared.coordinate_record import (
    BoundCoordinateRecord,
    bind_persisted_coordinate_record,
    stamp_and_bind_persisted_coordinate_record,
)
from fisheye.shared.run_provenance import (
    RUN_PROVENANCE_ATTR,
    build_run_provenance_from_stage_record,
    sha256_payload,
    validate_run_provenance,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.coordinate_reference import (
    bind_persisted_record_reference_extent,
    canonical_node_path,
)
from fisheye.shared.directed_transform_chain import (
    BoundDirectedTransformChain,
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_ATTR,
    DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
    DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    load_bound_directed_transform_v2,
    stamp_directed_transform_v2,
)
from fisheye.shared.pixel_frame_authority import (
    ARRAY_VALUES_CANONICALIZATION,
    CROP_PLACEMENT_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    BoundPixelFrameAuthority,
    load_crop_placement_ownership,
    load_roi_pixel_frame_authority,
    stamp_crop_placement_ownership,
    stamp_roi_pixel_frame_authority,
)
from fisheye.shared.proof_verification import (
    finish_proof_verification,
    proof_verification_operation,
)
from fisheye.shared.refined_subject_mask_mutation import (
    REFINED_SUBJECT_MASK_EDITABLE_DRAFT,
    refined_subject_mask_lifecycle_state,
    stamp_refined_subject_mask_editable_draft,
    stamp_refined_subject_mask_sealed_snapshot,
)
from fisheye.shared.subject_mask_coordinate_publication import (
    SUBJECT_MASK_COORDINATE_DERIVATION_ATTR,
    SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
    _load_subject_mask_coordinate_context,
    load_persisted_ineligible_subject_mask_coordinate_surfaces,
    load_persisted_subject_mask_coordinate_surfaces,
)
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    load_bound_transform_authority,
    stamp_crop_placement_transform_authority,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_CONTRACT,
    RUN_COMPLETION_CONTRACT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_COMPLETED_AT_ATTR,
    RUN_NAME_ATTR,
    RUN_STAGE_ATTR,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_RUNNING,
)
from fisheye.shared.zarr.coordinate_successor_authority import (
    REFINED_SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
    SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
    CoordinateSuccessorAuthorityError,
    load_coordinate_successor_authority,
    validate_coordinate_successor_authority,
)
from fisheye.shared.zarr.subject_mask_coordinate_validation_receipt import (
    REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
    SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE,
    SubjectMaskCoordinateValidationReceiptError,
    load_subject_mask_coordinate_validation_receipt,
)
from fisheye.shared.zarr.subject_mask_core_publication import (
    SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE,
    validate_subject_mask_core_run_manifest,
)
from fisheye.shared.zarr.subject_mask_bundle_publication import (
    SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID,
    validate_subject_mask_bundle_manifest,
)

REFINED_SUBJECT_MASK_COMPONENT_LABELS_ATTR = "refined_subject_mask_component_labels"
REFINED_SUBJECT_MASK_SOURCE_AUTHORITY_ATTR = "refined_subject_mask_source_authority"
REFINED_SUBJECT_MASK_REFINEMENT_AUTHORITY_ATTR = (
    "refined_subject_mask_refinement_authority"
)
REFINED_SUBJECT_MASK_COORDINATE_CONTEXT_ATTR = "refined_subject_mask_coordinate_context"
REFINED_SUBJECT_MASK_SURFACE_INVENTORY_ATTR = "refined_subject_mask_surface_inventory"
REFINED_SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR = (
    "refined_subject_mask_array_interpretation"
)
REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_ATTR = "refined_subject_mask_ragged_geometry"
REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR = (
    "refined_subject_mask_roi_reference_extent"
)
REFINED_SUBJECT_MASK_ASSIGNMENT_KEYPOINT_AUTHORITY_ATTR = (
    "refined_subject_mask_assignment_keypoint_authority"
)
REFINED_SUBJECT_MASK_COMPONENT_QC_INVENTORY_ATTR = (
    "refined_subject_mask_component_qc_inventory"
)
REFINED_SUBJECT_MASK_ACTIVATION_RECEIPT_ATTR = "refined_subject_mask_activation_receipt"
REFINED_SUBJECT_MASK_SCIENTIFIC_MANIFEST_ATTR = (
    "refined_subject_mask_scientific_manifest"
)
REFINED_SUBJECT_MASK_MEASUREMENT_AUTHORITY_ATTR = (
    "refined_subject_mask_measurement_authority"
)
REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR = "refined_subject_mask_publication_owner"
REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR = (
    "refined_subject_mask_publication_lease"
)
REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR = "publication_generation"
REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR = "publication_policy"

REFINED_SUBJECT_MASK_SCHEMA_VERSION = 1
REFINED_SUBJECT_MASK_COMPONENT_LABELS_SCHEMA_ID = (
    "palette.refined_subject_mask_component_labels"
)
REFINED_SUBJECT_MASK_SOURCE_AUTHORITY_SCHEMA_ID = (
    "palette.refined_subject_mask_source_authority"
)
REFINED_SUBJECT_MASK_REFINEMENT_AUTHORITY_SCHEMA_ID = (
    "palette.refined_subject_mask_refinement_authority"
)
REFINED_SUBJECT_MASK_COORDINATE_CONTEXT_SCHEMA_ID = (
    "palette.refined_subject_mask_coordinate_context"
)
REFINED_SUBJECT_MASK_SURFACE_INVENTORY_SCHEMA_ID = (
    "palette.refined_subject_mask_surface_inventory"
)
REFINED_SUBJECT_MASK_ARRAY_INTERPRETATION_SCHEMA_ID = (
    "palette.refined_subject_mask_array_interpretation"
)
REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_SCHEMA_ID = (
    "palette.refined_subject_mask_ragged_geometry"
)
REFINED_SUBJECT_MASK_REFERENCE_EXTENT_SCHEMA_ID = (
    "palette.refined_subject_mask_reference_extent"
)
REFINED_SUBJECT_MASK_ASSIGNMENT_KEYPOINT_AUTHORITY_SCHEMA_ID = (
    "palette.refined_subject_mask_assignment_keypoint_authority"
)
REFINED_SUBJECT_MASK_COMPONENT_QC_INVENTORY_SCHEMA_ID = (
    "palette.refined_subject_mask_component_qc_inventory"
)
REFINED_SUBJECT_MASK_ACTIVATION_RECEIPT_SCHEMA_ID = (
    "palette.refined_subject_mask_activation_receipt"
)
REFINED_SUBJECT_MASK_SCIENTIFIC_MANIFEST_SCHEMA_ID = (
    "palette.refined_subject_mask_scientific_manifest"
)
REFINED_SUBJECT_MASK_MEASUREMENT_AUTHORITY_SCHEMA_ID = (
    "palette.refined_subject_mask_measurement_authority"
)

_PUBLICATION_OWNER_RE = re.compile(r"^[0-9a-f]{32}$")
_PAYLOAD_SCAN_TARGET_BYTES = 8 * 1024 * 1024
_BOUND_CONTEXT_SEAL = object()
_BOUND_SURFACES_SEAL = object()
_CHECKPOINT_SEAL = object()
_PUBLICATION_POLICY = "owner_generation_guarded_selectors_then_eligibility_v1"
_ACTIVATION_GUARDED_SELECTOR_ATTRS = (
    "latest",
    "latest_complete",
    "authoritative_run",
    "authoritative_run_provenance",
)
_ACTIVATION_BASELINE_ATTRS = (
    *_ACTIVATION_GUARDED_SELECTOR_ATTRS,
    REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
    REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
)
_PUBLICATION_OWNED_ARRAY_ATTRS = frozenset(
    {
        "coordinate_descriptor",
        "coordinate_descriptor_sha256",
        "coordinate_descriptor_owner_dtype",
        REFINED_SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR,
        f"{REFINED_SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR}_sha256",
        REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_ATTR,
        f"{REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_ATTR}_sha256",
        ARRAY_MEASUREMENT_DESCRIPTOR_ATTR,
        f"{ARRAY_MEASUREMENT_DESCRIPTOR_ATTR}_sha256",
    }
)

_REFINED_COORDINATE_VALIDATION_RECORD_NAMES = frozenset(
    {
        "component_qc_inventory",
        "context",
        "measurement_authority",
        "refinement_authority",
        "row_identity",
        "scientific_manifest",
        "source_authority",
        "surface_inventory",
        "temporal_authority",
    }
)
_REFINED_COORDINATE_AUTHORITY_RECORD_NAMES = (
    _REFINED_COORDINATE_VALIDATION_RECORD_NAMES
    | {"coordinate_validation_receipt"}
)


@dataclass
class _PayloadCacheEntry:
    node: Any
    path: str
    shape: tuple[int, ...]
    dtype: str
    payload: dict[str, Any]


@dataclass
class _PayloadCacheState:
    entries: dict[tuple[int, str, tuple[int, ...], str], _PayloadCacheEntry] = field(
        default_factory=dict
    )
    lock: RLock = field(default_factory=RLock)


_PAYLOAD_CACHE: ContextVar[_PayloadCacheState | None] = ContextVar(
    "_REFINED_SUBJECT_MASK_PAYLOAD_CACHE",
    default=None,
)


@contextmanager
def _payload_cache_scope():
    """Memoize complete payload evidence for one refined publication call."""

    token = _PAYLOAD_CACHE.set(_PayloadCacheState())
    try:
        yield
    finally:
        _PAYLOAD_CACHE.reset(token)


_COMPONENT_METRIC_DEFINITIONS: Mapping[str, tuple[str, str, str, str]] = {
    "component_count": (
        "connected_component_count",
        "1",
        "connected_foreground_component_count_v1",
        "count",
    ),
    "largest_component_fraction": (
        "largest_component_area_fraction",
        "1",
        "largest_connected_component_fraction_v1",
        "fraction",
    ),
    "hole_count": (
        "enclosed_hole_count",
        "1",
        "enclosed_background_hole_count_v1",
        "count",
    ),
    "hole_area_fraction": (
        "enclosed_hole_area_fraction",
        "1",
        "enclosed_background_hole_fraction_v1",
        "fraction",
    ),
    "sigma_noise": (
        "contour_residual_scale",
        "px",
        "rms_contour_residual_from_smoothed_boundary_v1",
        "distance",
    ),
    "curvature_var": (
        "contour_curvature_variance",
        "px^-2",
        "variance_of_discrete_smoothed_contour_curvature_v1",
        "shape_metric",
    ),
    "ipr": (
        "isoperimetric_ratio",
        "1",
        "perimeter_squared_over_four_pi_area_v1",
        "shape_metric",
    ),
    "solidity": (
        "solidity",
        "1",
        "contour_area_over_convex_hull_area_v1",
        "fraction",
    ),
}
_FINALIZATION_METRIC_DEFINITIONS: Mapping[str, tuple[str, str, str, str]] = {
    "added_area_px": (
        "added_foreground_area",
        "px^2",
        "smart_finalizer_added_pixel_count_v1",
        "area",
    ),
    "area_px_after": (
        "foreground_area_after",
        "px^2",
        "smart_finalizer_output_pixel_count_v1",
        "area",
    ),
    "area_px_before": (
        "foreground_area_before",
        "px^2",
        "smart_finalizer_input_pixel_count_v1",
        "area",
    ),
    "changed_area_fraction": (
        "changed_area_fraction",
        "1",
        "smart_finalizer_changed_area_fraction_v1",
        "fraction",
    ),
    "changed_area_px": (
        "changed_foreground_area",
        "px^2",
        "smart_finalizer_changed_pixel_count_v1",
        "area",
    ),
    "component_count_after": (
        "connected_component_count_after",
        "1",
        "smart_finalizer_output_component_count_v1",
        "count",
    ),
    "component_count_before": (
        "connected_component_count_before",
        "1",
        "smart_finalizer_input_component_count_v1",
        "count",
    ),
    "hole_area_fraction_after": (
        "hole_area_fraction_after",
        "1",
        "smart_finalizer_output_hole_area_fraction_v1",
        "fraction",
    ),
    "hole_area_fraction_before": (
        "hole_area_fraction_before",
        "1",
        "smart_finalizer_input_hole_area_fraction_v1",
        "fraction",
    ),
    "hole_count_after": (
        "hole_count_after",
        "1",
        "smart_finalizer_output_hole_count_v1",
        "count",
    ),
    "hole_count_before": (
        "hole_count_before",
        "1",
        "smart_finalizer_input_hole_count_v1",
        "count",
    ),
    "largest_component_fraction_after": (
        "largest_component_fraction_after",
        "1",
        "smart_finalizer_output_largest_component_fraction_v1",
        "fraction",
    ),
    "largest_component_fraction_before": (
        "largest_component_fraction_before",
        "1",
        "smart_finalizer_input_largest_component_fraction_v1",
        "fraction",
    ),
    "removed_area_fraction": (
        "removed_area_fraction",
        "1",
        "smart_finalizer_removed_area_fraction_v1",
        "fraction",
    ),
    "removed_area_px": (
        "removed_foreground_area",
        "px^2",
        "smart_finalizer_removed_pixel_count_v1",
        "area",
    ),
    "removed_component_count": (
        "removed_component_count",
        "1",
        "smart_finalizer_removed_component_count_v1",
        "count",
    ),
    "removed_high_prob_area_px": (
        "removed_high_probability_area",
        "px^2",
        "smart_finalizer_removed_high_probability_pixel_count_v1",
        "area",
    ),
    "removed_prob_mass": (
        "removed_probability_mass",
        "px^2",
        "smart_finalizer_removed_probability_mass_v1",
        "area",
    ),
    "removed_prob_mass_fraction": (
        "removed_probability_mass_fraction",
        "1",
        "smart_finalizer_removed_probability_mass_fraction_v1",
        "fraction",
    ),
    "quality_code": (
        "review_quality_code",
        "1",
        "smart_finalizer_review_routing_code_v1",
        "categorical",
    ),
    "quality_score": (
        "review_quality_score",
        "1",
        "smart_finalizer_review_routing_score_v1",
        "score",
    ),
}


class RefinedSubjectMaskCoordinatePublicationError(ValueError):
    """Raised when refined geometry lacks exact future-normal authority."""


def _fail(message: str) -> None:
    raise RefinedSubjectMaskCoordinatePublicationError(message)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        _fail(f"Refined subject-mask metadata is not canonical JSON: {exc}.")


def _canonical_path(value: str, *, prefix: str, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} must be one canonical archive-relative path.")
    path = value.strip().strip("/")
    if (
        path != value
        or not path.startswith(prefix)
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        _fail(f"{label} path {value!r} is not canonical or uses the wrong parent.")
    return path


def _node(root: Any, path: str, *, label: str) -> Any:
    try:
        result = root[path]
    except Exception as exc:
        _fail(f"Persisted {label} is unavailable at {path!r}: {exc}.")
    if canonical_node_path(result) != path:
        _fail(f"Persisted {label} resolved to an unexpected path.")
    return result


def _child(group: Any, name: str, *, label: str) -> Any:
    base = canonical_node_path(group)
    try:
        result = group[name]
    except Exception as exc:
        _fail(f"{label} is unavailable: {exc}.")
    if canonical_node_path(result) != f"{base}/{name}":
        _fail(f"{label} resolved outside its canonical rowset.")
    return result


def _optional_child(group: Any, name: str) -> Any | None:
    try:
        if name not in group:
            return None
        return group[name]
    except Exception:
        return None


def _member_names(group: Any) -> tuple[str, ...]:
    """Return the exact direct child-name set for real or in-memory groups."""

    names: set[str] = set()
    for method_name in ("array_keys", "group_keys"):
        method = getattr(group, method_name, None)
        if callable(method):
            try:
                names.update(str(value) for value in method())
            except Exception as exc:
                _fail(f"Unable to enumerate {canonical_node_path(group)}: {exc}.")
    children = getattr(group, "children", None)
    if isinstance(children, Mapping):
        names.update(str(value) for value in children)
    if not names:
        keys = getattr(group, "keys", None)
        if callable(keys):
            try:
                names.update(str(value) for value in keys())
            except Exception as exc:
                _fail(f"Unable to enumerate {canonical_node_path(group)}: {exc}.")
    return tuple(sorted(names))


def _require_exact_members(
    group: Any,
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    actual = set(_member_names(group))
    wanted = set(expected)
    if actual != wanted:
        _fail(
            f"{label} is a closed-world geometry container; expected "
            f"{tuple(sorted(wanted))!r}, found {tuple(sorted(actual))!r}."
        )


def _exact_json_attrs(
    node: Any,
    *,
    label: str,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    excluded = set(exclude)
    attrs = copy.deepcopy(
        {
            str(name): value
            for name, value in dict(getattr(node, "attrs", {})).items()
            if str(name) not in excluded
        }
    )
    try:
        return json.loads(_canonical_json(attrs))
    except RefinedSubjectMaskCoordinatePublicationError:
        raise
    except Exception as exc:  # pragma: no cover - json implementation guard
        _fail(f"Unable to bind exact {label} attrs: {exc}.")


def _scientific_attrs(node: Any, *, label: str) -> dict[str, Any]:
    """Bind producer-owned attrs while excluding publication stamps themselves."""

    return _exact_json_attrs(
        node,
        label=label,
        exclude=tuple(_PUBLICATION_OWNED_ARRAY_ATTRS),
    )


def _recursive_namespace_inventory(node: Any, *, label: str) -> dict[str, Any]:
    """Bind every nested group attr and array payload in one typed namespace."""

    if hasattr(node, "shape"):
        return {
            "kind": "array",
            "payload": _payload(node),
            "producer_attrs": _scientific_attrs(node, label=label),
        }
    children = {
        name: _recursive_namespace_inventory(
            _child(node, name, label=f"{label}/{name}"),
            label=f"{label}/{name}",
        )
        for name in _member_names(node)
    }
    return {
        "kind": "group",
        "producer_attrs": _scientific_attrs(node, label=label),
        "children": children,
    }


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        value = np.asarray(node[:])
    except Exception as exc:
        _fail(f"Unable to read exact {label}: {exc}.")
    if value.dtype.hasobject:
        _fail(f"{label} cannot use object dtype.")
    return np.ascontiguousarray(value)


def _payload_digest_state(node: Any) -> tuple[Any, dict[str, Any]]:
    try:
        dtype = np.dtype(node.dtype)
        shape = tuple(int(item) for item in node.shape)
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(f"Coordinate surface lacks exact array metadata: {exc}.")
    metadata = {
        "array_ref": f"/{canonical_node_path(node)}",
        "shape": list(shape),
        "dtype": dtype.str,
    }
    header = {
        "canonicalization": ARRAY_VALUES_CANONICALIZATION,
        "dtype": np.lib.format.dtype_to_descr(dtype),
        "shape": list(shape),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(b"\x00")
    return digest, metadata


def _payload_row_chunk(node: Any) -> int:
    shape = tuple(int(item) for item in node.shape)
    if not shape or shape[0] <= 0:
        return 1
    inner = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    row_bytes = max(1, inner * int(np.dtype(node.dtype).itemsize))
    return max(1, min(shape[0], _PAYLOAD_SCAN_TARGET_BYTES // row_bytes))


def _payload_cache_key(
    node: Any,
    *,
    path: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> tuple[int, str, tuple[int, ...], str]:
    return id(node), path, shape, dtype.str


def _payload_cache_lookup(
    node: Any,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    state = _PAYLOAD_CACHE.get()
    if state is None:
        return None
    path = str(metadata["array_ref"]).removeprefix("/")
    shape = tuple(int(value) for value in metadata["shape"])
    dtype = np.dtype(str(metadata["dtype"]))
    key = _payload_cache_key(node, path=path, shape=shape, dtype=dtype)
    with state.lock:
        entry = state.entries.get(key)
        if entry is None:
            return None
        # Re-check all identity metadata on every hit.  The node reference in
        # the entry also prevents object-id reuse while this call is active.
        if (
            entry.node is not node
            or entry.path != path
            or entry.shape != shape
            or entry.dtype != dtype.str
        ):
            state.entries.pop(key, None)
            return None
        return copy.deepcopy(entry.payload)


def _payload_cache_store(
    node: Any,
    payload: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Store evidence only after the caller has completed its full scan."""

    state = _PAYLOAD_CACHE.get()
    if state is None:
        return
    if metadata is None:
        _, metadata = _payload_digest_state(node)
    path = str(metadata["array_ref"]).removeprefix("/")
    shape = tuple(int(value) for value in metadata["shape"])
    dtype = np.dtype(str(metadata["dtype"]))
    expected = {
        "array_ref": f"/{path}",
        "shape": list(shape),
        "dtype": dtype.str,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        _fail("Completed payload evidence metadata does not match its array node.")
    if not isinstance(payload.get("array_values_sha256"), str):
        _fail("Completed payload evidence lacks an exact values digest.")
    key = _payload_cache_key(node, path=path, shape=shape, dtype=dtype)
    entry = _PayloadCacheEntry(
        node=node,
        path=path,
        shape=shape,
        dtype=dtype.str,
        payload=copy.deepcopy(dict(payload)),
    )
    with state.lock:
        state.entries[key] = entry


def _payload(node: Any) -> dict[str, Any]:
    digest, metadata = _payload_digest_state(node)
    cached = _payload_cache_lookup(node, metadata=metadata)
    if cached is not None:
        return cached
    shape = tuple(int(item) for item in node.shape)
    label = canonical_node_path(node)
    if not shape:
        try:
            values = np.asarray(node[()])
        except Exception as exc:
            _fail(f"Unable to read exact {label}: {exc}.")
        if values.dtype != np.dtype(node.dtype):
            _fail(f"{label} changed dtype during payload validation.")
        digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    else:
        rows = _payload_row_chunk(node)
        for start in range(0, shape[0], rows):
            stop = min(shape[0], start + rows)
            try:
                values = np.asarray(node[start:stop])
            except Exception as exc:
                _fail(f"Unable to read exact {label} rows {start}:{stop}: {exc}.")
            if tuple(values.shape) != (stop - start, *shape[1:]):
                _fail(f"{label} changed shape during payload validation.")
            if values.dtype != np.dtype(node.dtype) or values.dtype.hasobject:
                _fail(f"{label} changed dtype during payload validation.")
            digest.update(np.ascontiguousarray(values).tobytes(order="C"))
    payload = {**metadata, "array_values_sha256": digest.hexdigest()}
    _payload_cache_store(node, payload, metadata=metadata)
    return payload


def _record_pointer(value: BoundCoordinateRecord) -> dict[str, str]:
    return {
        "record_ref": value.record_ref,
        "record_sha256": value.record_sha256,
    }


def _publication_owner(run: Any, *, expected: str | None = None) -> str:
    value = getattr(run, "attrs", {}).get(REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR)
    if not isinstance(value, str) or _PUBLICATION_OWNER_RE.fullmatch(value) is None:
        _fail(
            "Canonical refined subject-mask run lacks one unguessable publication owner."
        )
    if expected is not None and value != expected:
        _fail("Canonical refined subject-mask run was replaced by another owner.")
    return value


def _require_run_status(
    run: Any,
    *,
    status: str,
    selector_eligible: bool,
    label: str,
) -> None:
    attrs = getattr(run, "attrs", {})
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != status
        or attrs.get("stage_selector_eligible") is not selector_eligible
    ):
        _fail(
            f"{label} must carry the exact completion contract with status "
            f"{status!r} and selector eligibility {selector_eligible!r}."
        )


def _fresh_owned_ineligible_run(
    root: Any,
    path: str,
    *,
    owner: str,
    statuses: Sequence[str],
    label: str,
) -> Any:
    run = _node(root, path, label=label)
    _publication_owner(run, expected=owner)
    attrs = getattr(run, "attrs", {})
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) not in set(statuses)
        or attrs.get("stage_selector_eligible") is not False
    ):
        _fail(f"{label} must remain the exact owned selector-ineligible child.")
    return run


def _require_group(parent: Any, name: str) -> tuple[Any, bool]:
    try:
        if name in parent:
            node = parent[name]
            if hasattr(node, "shape"):
                _fail(f"Expected group {name!r}, found an array.")
            return node, False
    except TypeError:
        pass
    create = getattr(parent, "create_group", None)
    if not callable(create):
        _fail(f"Cannot create required coordinate group {name!r}.")
    return create(name), True


def _attrs_snapshot(*nodes: Any) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
    unique: list[Any] = []
    seen: set[int] = set()
    for node in nodes:
        if id(node) in seen:
            continue
        seen.add(id(node))
        attrs = getattr(node, "attrs", None)
        if attrs is None or not hasattr(attrs, "keys"):
            _fail("Coordinate publication target lacks mutable attrs.")
        unique.append(node)
    return (
        tuple(canonical_node_path(node) for node in unique),
        tuple(copy.deepcopy(dict(node.attrs)) for node in unique),
    )


def _restore_attrs(
    root: Any,
    paths: Sequence[str],
    snapshots: Sequence[Mapping[str, Any]],
    *,
    run_path: str,
    owner: str,
) -> None:
    failures: list[str] = []
    for path, snapshot in zip(paths, snapshots, strict=True):
        try:
            _fresh_owned_ineligible_run(
                root,
                run_path,
                owner=owner,
                statuses=(RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE),
                label="Refined coordinate rollback target",
            )
            node = _node(root, path, label="refined coordinate rollback node")
            for name in tuple(node.attrs.keys()):
                del node.attrs[name]
            node.attrs.update(copy.deepcopy(dict(snapshot)))
            if dict(node.attrs) != dict(snapshot):
                raise RuntimeError("restored attrs differ from exact snapshot")
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{path}: {exc}")
    if failures:
        raise RuntimeError(
            f"Refined coordinate attrs rollback was incomplete: {failures!r}."
        )


def _delete_created(
    root: Any,
    paths: Sequence[str],
    *,
    run_path: str,
    owner: str,
) -> None:
    for path in reversed(tuple(paths)):
        try:
            _fresh_owned_ineligible_run(
                root,
                run_path,
                owner=owner,
                statuses=(RUN_STATUS_RUNNING,),
                label="Refined coordinate rollback target",
            )
            del root[path]
        except BaseException:
            pass
    survivors: list[str] = []
    for path in paths:
        try:
            root[path]
        except BaseException:
            continue
        survivors.append(path)
    if survivors:
        raise RuntimeError(
            f"Created coordinate nodes survived rollback: {survivors!r}."
        )


def _labels(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        _fail("Refined component labels must be a nonempty sequence.")
    labels = tuple(str(item) for item in value)
    if (
        not labels
        or len(labels) != len(set(labels))
        or any(not item or item != item.strip() for item in labels)
    ):
        _fail("Refined component labels must be unique canonical strings.")
    return labels


def _source_metadata_context(
    root: Any,
    source_path: str,
    *,
    selector_eligible: bool = True,
) -> Any:
    """Load exact raw coordinate metadata without scanning its large payload."""

    try:
        return _load_subject_mask_coordinate_context(
            root,
            source_path,
            require_complete=True,
            expected_selector_eligible=selector_eligible,
        )
    except Exception as exc:
        raise RefinedSubjectMaskCoordinatePublicationError(
            f"Selected raw subject-mask coordinate context is invalid: {exc}"
        ) from exc


def _validate_exact_source_selection(source: Any, run: Any) -> dict[str, Any]:
    source_run = source._run_group
    names = (
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "source_crop_xywh",
    )
    result: dict[str, Any] = {}
    for name in names:
        source_node = _child(source_run, name, label=f"raw {name}")
        target_node = _child(run, name, label=f"refined {name}")
        source_values = _array(source_node, label=f"raw {name}")
        target_values = _array(target_node, label=f"refined {name}")
        if source_values.dtype != target_values.dtype or not np.array_equal(
            source_values,
            target_values,
        ):
            _fail(
                f"Refined {name} is not an exact dtype-preserving copy from "
                "the selected raw subject-mask run."
            )
        result[name] = target_node
    forbidden = tuple(
        name
        for name in (
            "frame_indices",
            "frame_counts",
            "detection_indices",
            "source_frame_indices",
            "source_refined_row_ids",
            "source_detect_row_index",
        )
        if name in run
    )
    if forbidden:
        _fail(
            "Future refined rows must not add ambiguous legacy row aliases; "
            f"found {forbidden!r}."
        )
    return result


def _label_record(labels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_id": REFINED_SUBJECT_MASK_COMPONENT_LABELS_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "axis": 1,
        "role": "subject_component",
        "cardinality": len(labels),
        "labels": list(labels),
    }


def _assignment_keypoint_record(
    source: Any,
    run: Any,
    surfaces: BoundKeypointCoordinateSurfaces | None,
) -> dict[str, Any]:
    assignment_contract = run.attrs.get("assignment_keypoint_coordinate_contract")
    if surfaces is None:
        if assignment_contract is not None:
            _fail(
                "Refined run declares canonical assignment keypoints without a sealed dependency."
            )
        return {
            "schema_id": REFINED_SUBJECT_MASK_ASSIGNMENT_KEYPOINT_AUTHORITY_SCHEMA_ID,
            "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
            "status": "not_used",
            "reason": "no_canonical_keypoint_derived_eye_assignment",
        }

    try:
        surfaces = require_bound_keypoint_coordinate_surfaces(surfaces)
    except Exception as exc:
        raise RefinedSubjectMaskCoordinatePublicationError(
            f"Canonical assignment keypoint dependency is stale or invalid: {exc}"
        ) from exc
    context = surfaces.context
    if not context.run_path.startswith("keypoints_runs/"):
        _fail("Canonical refined eye assignment accepts raw keypoints_runs only.")
    if assignment_contract != "canonical_v2_exact":
        _fail(
            "Refined eye assignment lacks exact canonical keypoint contract metadata."
        )
    for name, expected in (
        ("assignment_keypoint_coordinate_run_path", context.run_path),
        ("assignment_keypoint_group", "keypoints_runs"),
        ("assignment_keypoints_run", context.run_path.split("/", 1)[1]),
    ):
        if run.attrs.get(name) != expected:
            _fail(
                f"Refined eye assignment {name} differs from its sealed keypoint source."
            )
    if context.source.crop_path != source.source.crop_path:
        _fail(
            "Assignment keypoints and raw subject masks use different exact crop runs."
        )
    source_run = source._run_group
    keypoint_run = context._run_group
    source_attrs = source_run.attrs
    assignment_pair = (
        source_attrs.get("assignment_keypoint_group"),
        source_attrs.get("assignment_keypoints_run"),
    )
    source_pair = (
        source_attrs.get("source_keypoint_group"),
        source_attrs.get("source_keypoints_run"),
    )
    if bool(assignment_pair[0]) != bool(assignment_pair[1]):
        _fail("Canonical raw subject-mask source has incomplete assignment_* lineage.")
    if bool(source_pair[0]) != bool(source_pair[1]):
        _fail(
            "Canonical raw subject-mask source has incomplete source_* keypoint lineage."
        )
    if (
        all(assignment_pair)
        and all(source_pair)
        and tuple(map(str, assignment_pair)) != tuple(map(str, source_pair))
    ):
        _fail(
            "Canonical raw subject-mask source has conflicting complete assignment_* "
            "and source_* keypoint lineage."
        )
    selection_payloads: dict[str, dict[str, Any]] = {}
    for name in (
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
        "source_crop_xywh",
    ):
        mask_node = _child(source_run, name, label=f"raw subject-mask {name}")
        keypoint_node = _child(keypoint_run, name, label=f"assignment keypoint {name}")
        mask_values = _array(mask_node, label=f"raw subject-mask {name}")
        keypoint_values = _array(keypoint_node, label=f"assignment keypoint {name}")
        if (
            mask_values.dtype != keypoint_values.dtype
            or mask_values.shape != keypoint_values.shape
            or not np.array_equal(mask_values, keypoint_values)
        ):
            _fail(f"Assignment keypoints do not exactly match raw subject-mask {name}.")
        selection_payloads[name] = {
            "subject_mask": _payload(mask_node),
            "keypoints": _payload(keypoint_node),
        }
    source_extent = (
        int(source.continuous_frame.endpoint.height),
        int(source.continuous_frame.endpoint.width),
    )
    keypoint_extent = (
        int(context.roi_frame.endpoint.height),
        int(context.roi_frame.endpoint.width),
    )
    if source_extent != keypoint_extent:
        _fail(
            "Assignment keypoints and raw subject masks use different exact ROI extents."
        )

    success_name = run.attrs.get("assignment_keypoint_success_dataset")
    if success_name != "detection_success":
        _fail(
            "Canonical eye assignment must name the exact raw-keypoint "
            "detection_success leaf."
        )
    success_node = _child(
        keypoint_run,
        success_name,
        label="assignment keypoint success",
    )
    if tuple(int(value) for value in success_node.shape) != (
        context.row_identity.leading_dimension,
    ) or np.dtype(success_node.dtype) != np.dtype("bool"):
        _fail("Canonical keypoint success must be exact bool shape (N,).")
    labels_raw = keypoint_run.attrs.get("keypoint_labels")
    if not isinstance(labels_raw, (list, tuple)) or not labels_raw:
        _fail("Canonical assignment keypoints lack exact keypoint_labels.")
    keypoint_labels = [str(value) for value in labels_raw]
    indices = run.attrs.get("assignment_keypoint_eye_indices")
    if type(indices) is not dict or set(indices) != {"eye_left", "eye_right"}:
        _fail("Canonical eye assignment lacks exact anatomical keypoint indices.")
    normalized_indices: dict[str, int] = {}
    for name in ("eye_left", "eye_right"):
        value = indices.get(name)
        if type(value) is not int or value < 0 or value >= len(keypoint_labels):
            _fail("Canonical eye-assignment keypoint index is invalid.")
        label = keypoint_labels[value].strip().lower()
        accepted = {
            "eye_left": {"eye_left", "left", "left_eye"},
            "eye_right": {"eye_right", "right", "right_eye"},
        }[name]
        if label not in accepted:
            _fail("Canonical eye-assignment indices disagree with keypoint_labels.")
        normalized_indices[name] = value
    if normalized_indices["eye_left"] == normalized_indices["eye_right"]:
        _fail("Canonical eye-assignment indices must identify distinct keypoints.")

    descriptor_ref = f"/{context.run_path}/keypoints_roi@coordinate_descriptor"
    descriptor_sha = surfaces.keypoints_roi.descriptor.digest()
    if (
        run.attrs.get("assignment_keypoint_roi_descriptor_ref") != descriptor_ref
        or run.attrs.get("assignment_keypoint_roi_descriptor_sha256") != descriptor_sha
        or run.attrs.get("assignment_keypoint_coordinate_derivation_ref")
        != surfaces.derivation.record_ref
        or run.attrs.get("assignment_keypoint_coordinate_derivation_sha256")
        != surfaces.derivation.record_sha256
        or run.attrs.get("assignment_keypoint_row_identity_ref")
        != context.row_identity.record_ref
        or run.attrs.get("assignment_keypoint_row_identity_sha256")
        != context.row_identity.record_sha256
    ):
        _fail("Refined keypoint dependency pointers differ from exact live authority.")
    raw_inventory = bind_persisted_coordinate_record(
        source_run,
        attr_name=SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
    )
    raw_derivation = bind_persisted_coordinate_record(
        source_run,
        attr_name=SUBJECT_MASK_COORDINATE_DERIVATION_ATTR,
    )
    raw_geometry = raw_inventory.record.get("geometry")
    if not isinstance(raw_geometry, dict):
        _fail("Canonical raw subject-mask inventory lacks geometry surfaces.")
    raw_surface_name = "masks_roi" if "masks_roi" in raw_geometry else "mask_probs_roi"
    raw_surface = raw_geometry.get(raw_surface_name)
    if not isinstance(raw_surface, dict) or not isinstance(
        raw_surface.get("payload"), dict
    ):
        _fail(
            "Canonical raw subject-mask inventory lacks an authoritative mask payload."
        )
    return {
        "schema_id": REFINED_SUBJECT_MASK_ASSIGNMENT_KEYPOINT_AUTHORITY_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "status": "used",
        "selection_policy": "exact_full_raw_keypoint_rowset_no_fallback_v1",
        "keypoint_run_path": context.run_path,
        "crop_run_path": context.source.crop_path,
        "keypoint_labels": keypoint_labels,
        "eye_keypoint_indices": normalized_indices,
        "keypoints_roi": {
            "payload": _payload(surfaces.keypoints_roi.coordinate_node),
            "descriptor_ref": descriptor_ref,
            "descriptor_sha256": descriptor_sha,
        },
        "success": {
            "dataset": success_name,
            "payload": _payload(success_node),
        },
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "coordinate_context": _record_pointer(context.context_record),
        "coordinate_derivation": _record_pointer(surfaces.derivation),
        "raw_subject_mask_dependency": {
            "run_path": source.run_path,
            "surface_role": raw_surface_name,
            "surface_payload": copy.deepcopy(raw_surface["payload"]),
            "surface_inventory": _record_pointer(raw_inventory),
            "coordinate_derivation": _record_pointer(raw_derivation),
            "row_identity": {
                "record_ref": source.row_identity.record_ref,
                "record_sha256": source.row_identity.record_sha256,
            },
        },
        "selection_compatibility": selection_payloads,
        "roi_extent": {
            "width": keypoint_extent[1],
            "height": keypoint_extent[0],
            "units": "px",
        },
    }


def _assignment_dependency_summary(
    authority: BoundCoordinateRecord,
) -> dict[str, Any]:
    record = authority.record
    result: dict[str, Any] = {
        "status": record.get("status"),
        "authority": _record_pointer(authority),
    }
    if record.get("status") == "used":
        for name in (
            "keypoint_run_path",
            "crop_run_path",
            "keypoints_roi",
            "success",
            "row_identity",
            "coordinate_context",
            "coordinate_derivation",
            "raw_subject_mask_dependency",
            "selection_policy",
            "roi_extent",
        ):
            result[name] = copy.deepcopy(record[name])
    else:
        result["reason"] = record.get("reason")
    return result


def _component_source_selection_records(
    source: Any,
    run: Any,
    labels: tuple[str, ...],
    *,
    raw_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact producer-selected raw surface for every output component."""

    source_labels = tuple(str(value) for value in source.labels)
    source_run = source._run_group
    expected_probability_encoding = source_run.attrs.get("probabilities_encoding")
    expected_probability_threshold = source.inference_authority.record.get(
        "mask_probability_threshold"
    )
    components = _child(run, "components", label="refined components")
    result: dict[str, Any] = {}
    for component in labels:
        component_group = _child(
            components,
            component,
            label=f"refined component {component}",
        )
        provenance = _child(
            component_group,
            "provenance",
            label=f"refined component {component} provenance",
        )
        if hasattr(provenance, "shape"):
            _fail(f"Refined component {component!r} provenance must be a group.")
        attrs = _exact_json_attrs(
            provenance,
            label=f"refined component {component} provenance",
        )
        channels_raw = attrs.get("source_channels")
        if type(channels_raw) is not list or len(channels_raw) != 1:
            _fail(
                f"Refined component {component!r} must name exactly one selected raw channel."
            )
        source_channel = str(channels_raw[0])
        if source_channel not in source_labels:
            _fail(
                f"Refined component {component!r} selects an unknown raw component "
                f"{source_channel!r}."
            )
        source_component_index = source_labels.index(source_channel)
        source_surface_path = attrs.get("source_surface_path")
        if type(source_surface_path) is not str:
            _fail(f"Refined component {component!r} lacks exact source_surface_path.")
        expected_prefix = f"{source.run_path}/"
        if not source_surface_path.startswith(expected_prefix):
            _fail(
                f"Refined component {component!r} source surface leaves the selected raw run."
            )
        source_surface_name = source_surface_path[len(expected_prefix) :]
        if "/" in source_surface_name or source_surface_name not in {
            "mask_probs_roi",
            "masks_roi",
        }:
            _fail(
                f"Refined component {component!r} names unsupported source surface "
                f"{source_surface_name!r}."
            )
        source_surface = raw_geometry.get(source_surface_name)
        if not isinstance(source_surface, Mapping) or not isinstance(
            source_surface.get("payload"),
            Mapping,
        ):
            _fail(
                f"Refined component {component!r} source surface is absent from the raw inventory."
            )
        source_kind = attrs.get("source_surface_kind")
        expected_kind = (
            "probability" if source_surface_name == "mask_probs_roi" else "binary"
        )
        if source_kind != expected_kind:
            _fail(
                f"Refined component {component!r} source kind contradicts its exact surface."
            )
        selection: dict[str, Any] = {
            "refined_component": component,
            "source_component": source_channel,
            "source_component_index": source_component_index,
            "source_surface_path": source_surface_path,
            "source_surface_role": source_surface_name,
            "source_surface_kind": expected_kind,
            "source_surface_payload": copy.deepcopy(dict(source_surface["payload"])),
            "source_surface_interpretation": copy.deepcopy(
                source_surface.get("interpretation")
            ),
            "finalization_method": attrs.get("finalization_method"),
            "finalization_policy": copy.deepcopy(attrs.get("finalization_policy")),
            "component_provenance_path": canonical_node_path(provenance),
            "component_provenance_attrs": attrs,
        }
        if (
            type(selection["finalization_method"]) is not str
            or not selection["finalization_method"]
            or type(selection["finalization_policy"]) is not dict
        ):
            _fail(
                f"Refined component {component!r} lacks exact finalization method/policy."
            )
        if expected_kind == "probability":
            encoding = attrs.get("source_probability_encoding")
            threshold = attrs.get("source_probability_threshold")
            if (
                type(encoding) is not str
                or not encoding
                or encoding != expected_probability_encoding
                or type(threshold) is not float
                or not np.isfinite(threshold)
                or threshold != expected_probability_threshold
                or attrs.get("source_probability_path") != source_surface_path
                or attrs.get("source_binary_derivation")
                != "smart_finalize(mask_probs_roi)"
            ):
                _fail(
                    f"Refined component {component!r} probability encoding, threshold, "
                    "path, or binary derivation differs from the exact selected raw surface."
                )
            selection["probability_encoding"] = encoding
            selection["probability_threshold"] = threshold
        elif attrs.get("source_binary_derivation") != "smart_finalize(masks_roi)":
            _fail(f"Refined component {component!r} binary derivation is not exact.")
        result[component] = selection
    return result


def _source_authority_record(
    source: Any,
    run: Any,
    labels: tuple[str, ...],
    *,
    assignment_keypoints: BoundCoordinateRecord,
) -> dict[str, Any]:
    raw_run = source._run_group
    provenance = raw_run.attrs.get("provenance")
    if type(provenance) is not dict:
        _fail("Canonical raw subject-mask source lacks exact stage provenance.")
    inventory = bind_persisted_coordinate_record(
        raw_run,
        attr_name=SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
    )
    derivation = bind_persisted_coordinate_record(
        raw_run,
        attr_name=SUBJECT_MASK_COORDINATE_DERIVATION_ATTR,
    )
    geometry = inventory.record.get("geometry")
    if not isinstance(geometry, dict):
        _fail("Raw subject-mask inventory lacks exact geometry payloads.")
    component_selections = _component_source_selection_records(
        source,
        run,
        labels,
        raw_geometry=geometry,
    )
    return {
        "schema_id": REFINED_SUBJECT_MASK_SOURCE_AUTHORITY_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "source_run_path": source.run_path,
        "source_coordinate_context": _record_pointer(source.context_record),
        "source_surface_inventory": _record_pointer(inventory),
        "source_coordinate_derivation": _record_pointer(derivation),
        "component_source_selections": component_selections,
        "source_inference_authority": _record_pointer(source.inference_authority),
        "source_stage_provenance": copy.deepcopy(provenance),
        "source_stage_provenance_sha256": hashlib.sha256(
            _canonical_json(provenance).encode("utf-8")
        ).hexdigest(),
        "source_row_identity": {
            "record_ref": source.row_identity.record_ref,
            "record_sha256": source.row_identity.record_sha256,
        },
        "selection_policy": "exact_full_raw_subject_mask_rowset_v1",
        "assignment_keypoint_dependency": _assignment_dependency_summary(
            assignment_keypoints
        ),
    }


def _refinement_authority_record(
    run: Any,
    *,
    source_authority: BoundCoordinateRecord,
    assignment_keypoints: BoundCoordinateRecord,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    attrs = getattr(run, "attrs", {})
    provenance = attrs.get("provenance")
    if type(provenance) is not dict:
        _fail("Canonical refined runs require exact persisted stage provenance.")
    method = attrs.get("method")
    refinement_semantics = attrs.get("refinement_semantics")
    finalization_semantics = attrs.get("finalization_semantics")
    if not all(
        isinstance(item, str) and item.strip() == item and item
        for item in (
            method,
            refinement_semantics,
            finalization_semantics,
        )
    ):
        _fail("Canonical refined runs require explicit method/refinement semantics.")
    return {
        "schema_id": REFINED_SUBJECT_MASK_REFINEMENT_AUTHORITY_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "method": method,
        "refinement_semantics": refinement_semantics,
        "finalization_semantics": finalization_semantics,
        "label_schema_id": attrs.get("label_schema_id"),
        "mask_labels": list(labels),
        "source_authority": _record_pointer(source_authority),
        "assignment_keypoint_dependency": _assignment_dependency_summary(
            assignment_keypoints
        ),
        "stage_provenance": copy.deepcopy(provenance),
        "stage_provenance_sha256": hashlib.sha256(
            _canonical_json(provenance).encode("utf-8")
        ).hexdigest(),
    }


def _activation_receipt_record(
    run: Any,
    *,
    refinement_authority: BoundCoordinateRecord,
) -> dict[str, Any]:
    """Build the deterministic post-completion activation receipt."""

    attrs = run.attrs
    run_path = canonical_node_path(run)
    run_name = run_path.split("/", 1)[1] if "/" in run_path else ""
    if (
        attrs.get(RUN_COMPLETION_CONTRACT_ATTR) != RUN_COMPLETION_CONTRACT
        or attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE
        or attrs.get(RUN_NAME_ATTR) != run_name
        or not isinstance(attrs.get(RUN_COMPLETED_AT_ATTR), str)
        or not attrs.get(RUN_COMPLETED_AT_ATTR)
        or not isinstance(attrs.get(RUN_STAGE_ATTR), str)
        or not attrs.get(RUN_STAGE_ATTR)
    ):
        _fail("Canonical refined activation requires exact complete-run metadata.")
    stage_provenance = refinement_authority.record.get("stage_provenance")
    if not isinstance(stage_provenance, dict):
        _fail("Refined activation receipt lacks sealed stage provenance.")
    live_stage = attrs.get("provenance")
    if live_stage != stage_provenance:
        _fail("Refined completion stage provenance contradicts its sealed authority.")
    run_provenance = attrs.get(RUN_PROVENANCE_ATTR)
    validation = validate_run_provenance(run_provenance)
    if not validation.valid or validation.normalized is None:
        _fail(
            "Canonical refined activation requires valid persisted run_provenance: "
            + "; ".join(validation.errors)
        )
    normalized = validation.normalized
    expected_command = str(stage_provenance.get("command") or "finalize_subject_masks")
    expected_params = (
        dict(stage_provenance["parameters"])
        if isinstance(stage_provenance.get("parameters"), Mapping)
        else {}
    )
    expected_inputs = (
        dict(stage_provenance["inputs"])
        if isinstance(stage_provenance.get("inputs"), Mapping)
        else {}
    )
    if (
        normalized.get("command") != expected_command
        or normalized.get("params") != expected_params
        or normalized.get("input_run_ids") != expected_inputs
        or normalized.get("config_hash") != sha256_payload(expected_params)
    ):
        _fail(
            "Refined run_provenance is not mechanically derived from the sealed "
            "stage provenance."
        )
    completion = {
        name: copy.deepcopy(attrs.get(name))
        for name in (
            RUN_COMPLETION_CONTRACT_ATTR,
            RUN_COMPLETION_STATUS_ATTR,
            RUN_COMPLETED_AT_ATTR,
            RUN_NAME_ATTR,
            RUN_STAGE_ATTR,
        )
    }
    return {
        "schema_id": REFINED_SUBJECT_MASK_ACTIVATION_RECEIPT_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "run_path": run_path,
        "completion": completion,
        "refinement_authority": _record_pointer(refinement_authority),
        "stage_provenance_sha256": hashlib.sha256(
            _canonical_json(stage_provenance).encode("utf-8")
        ).hexdigest(),
        "run_provenance": copy.deepcopy(normalized),
        "run_provenance_sha256": hashlib.sha256(
            _canonical_json(normalized).encode("utf-8")
        ).hexdigest(),
        "linkage_policy": "exact_stage_parameters_inputs_command_to_run_provenance_v1",
    }


def _stamp_refined_subject_mask_activation_receipt(
    root: Any,
    run_path: str,
    *,
    owner: str,
) -> BoundCoordinateRecord:
    """Seal completion evidence after ``mark_run_complete`` and before selectors."""

    run = _fresh_owned_ineligible_run(
        root,
        run_path,
        owner=owner,
        statuses=(RUN_STATUS_COMPLETE,),
        label="Refined activation receipt target",
    )
    refinement_authority = bind_persisted_coordinate_record(
        run,
        attr_name=REFINED_SUBJECT_MASK_REFINEMENT_AUTHORITY_ATTR,
    )
    if RUN_PROVENANCE_ATTR not in run.attrs:
        stage_provenance = refinement_authority.record.get("stage_provenance")
        if not isinstance(stage_provenance, dict):
            _fail(
                "Cannot derive refined run provenance without sealed stage provenance."
            )
        run.attrs[RUN_PROVENANCE_ATTR] = build_run_provenance_from_stage_record(
            stage_provenance,
            fallback_command="finalize_subject_masks",
        )
        run = _fresh_owned_ineligible_run(
            root,
            run_path,
            owner=owner,
            statuses=(RUN_STATUS_COMPLETE,),
            label="Refined activation receipt fresh target",
        )
        refinement_authority = bind_persisted_coordinate_record(
            run,
            attr_name=REFINED_SUBJECT_MASK_REFINEMENT_AUTHORITY_ATTR,
        )
    record = _activation_receipt_record(
        run,
        refinement_authority=refinement_authority,
    )
    return stamp_and_bind_persisted_coordinate_record(
        run,
        record,
        attr_name=REFINED_SUBJECT_MASK_ACTIVATION_RECEIPT_ATTR,
    )


def _extent_record(
    *,
    width: int,
    height: int,
    convention: str,
    source_frame: BoundPixelFrameAuthority,
    source_rows_node: Any,
) -> dict[str, Any]:
    return {
        "schema_id": REFINED_SUBJECT_MASK_REFERENCE_EXTENT_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "role": "refined_subject_mask_native_roi",
        "operation": "exact_selected_raw_roi_extent_copy_v1",
        "pixel_convention": convention,
        "width": int(width),
        "height": int(height),
        "units": "px",
        "source_raw_roi_frame": {
            "record_ref": source_frame.record_ref,
            "record_sha256": source_frame.record_sha256,
        },
        "source_crop_row_ids": _payload(source_rows_node),
    }


def _stamp_extent(node: Any, record: Mapping[str, Any]) -> Any:
    for name in ("width", "height"):
        expected = record[name]
        if name in node.attrs and node.attrs[name] != expected:
            _fail(f"Existing refined ROI {name} conflicts with exact source extent.")
        node.attrs[name] = expected
    stamp_and_bind_persisted_coordinate_record(
        node,
        dict(record),
        attr_name=REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR,
    )
    return bind_persisted_record_reference_extent(
        node,
        record_attr=REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR,
        digest_attr=f"{REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR}_sha256",
        width_field="width",
        height_field="height",
        units_field="units",
    )


def _context_record(
    *,
    run_path: str,
    source: Any,
    identity: BoundRowIdentityContract,
    temporal: BoundSourceRowTemporalAuthority,
    labels: BoundCoordinateRecord,
    assignment_keypoints: BoundCoordinateRecord,
    source_authority: BoundCoordinateRecord,
    refinement_authority: BoundCoordinateRecord,
    continuous_frame: BoundPixelFrameAuthority,
    continuous_chain: BoundDirectedTransformChain,
    center_frame: BoundPixelFrameAuthority,
    center_chain: BoundDirectedTransformChain,
    edge_frame: BoundPixelFrameAuthority,
    edge_chain: BoundDirectedTransformChain,
    selection: Mapping[str, Any],
    owner: str,
    source_selector_eligible: bool = True,
) -> dict[str, Any]:
    record = {
        "schema_id": REFINED_SUBJECT_MASK_COORDINATE_CONTEXT_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "run_path": run_path,
        "publication_owner": owner,
        "source_subject_mask_path": source.run_path,
        "row_identity": {
            "record_ref": identity.record_ref,
            "record_sha256": identity.record_sha256,
        },
        "temporal_authority": {
            "record_ref": temporal.record_ref,
            "record_sha256": temporal.record_sha256,
        },
        "component_labels": _record_pointer(labels),
        "assignment_keypoint_dependency": _assignment_dependency_summary(
            assignment_keypoints
        ),
        "source_authority": _record_pointer(source_authority),
        "refinement_authority": _record_pointer(refinement_authority),
        "selection": {name: _payload(node) for name, node in selection.items()},
        "roi_frames": {
            "continuous": {
                "record_ref": continuous_frame.record_ref,
                "record_sha256": continuous_frame.record_sha256,
            },
            "pixel_center": {
                "record_ref": center_frame.record_ref,
                "record_sha256": center_frame.record_sha256,
            },
            "pixel_edge_half_open": {
                "record_ref": edge_frame.record_ref,
                "record_sha256": edge_frame.record_sha256,
            },
        },
        "roi_to_source_camera": {
            "direction": "roi_local_px_to_source_camera_image_px",
            "continuous": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in continuous_chain.transform_records
            ],
            "pixel_center": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in center_chain.transform_records
            ],
            "pixel_edge_half_open": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in edge_chain.transform_records
            ],
        },
        "raster_authority": {
            "surface": "masks_roi",
            "coordinate_space": "roi_local_px",
            "compact_cache_policy": "derived_non_authoritative",
        },
    }
    if source_selector_eligible is not True:
        record["source_selector_eligible"] = False
    return record


@dataclass(frozen=True, init=False)
class BoundRefinedSubjectMaskCoordinateContext:
    source: Any = field(repr=False)
    row_identity: BoundRowIdentityContract = field(repr=False)
    temporal_authority: BoundSourceRowTemporalAuthority = field(repr=False)
    component_labels: BoundCoordinateRecord = field(repr=False)
    assignment_keypoint_authority: BoundCoordinateRecord = field(repr=False)
    assignment_keypoint_surfaces: BoundKeypointCoordinateSurfaces | None = field(
        repr=False
    )
    source_authority: BoundCoordinateRecord = field(repr=False)
    refinement_authority: BoundCoordinateRecord = field(repr=False)
    labels: tuple[str, ...]
    continuous_frame: BoundPixelFrameAuthority = field(repr=False)
    continuous_chain: BoundDirectedTransformChain = field(repr=False)
    pixel_center_frame: BoundPixelFrameAuthority = field(repr=False)
    pixel_center_chain: BoundDirectedTransformChain = field(repr=False)
    pixel_edge_frame: BoundPixelFrameAuthority = field(repr=False)
    pixel_edge_chain: BoundDirectedTransformChain = field(repr=False)
    context_record: BoundCoordinateRecord = field(repr=False)
    run_path: str
    completion_status: str
    selector_eligible: bool
    publication_owner: str
    _root: Any = field(repr=False, compare=False)
    _run_group: Any = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self, *, _verification_seal: object | None = None, **values: Any
    ) -> None:
        if _verification_seal is not _BOUND_CONTEXT_SEAL:
            _fail("Refined coordinate contexts cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


@proof_verification_operation
def prepare_refined_subject_mask_coordinate_context(
    root: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
    source_subject_mask_path: str,
    mask_labels: Sequence[str],
    assignment_keypoint_surfaces: BoundKeypointCoordinateSurfaces | None = None,
    source_selector_eligible: bool = True,
) -> BoundRefinedSubjectMaskCoordinateContext:
    """Bind one running refined child to its exact raw coordinate authority."""

    path = _canonical_path(
        run_path,
        prefix="refined_subject_masks_runs/",
        label="refined subject-mask rowset",
    )
    run = _fresh_owned_ineligible_run(
        root,
        path,
        owner=expected_publication_owner,
        statuses=(RUN_STATUS_RUNNING,),
        label="Refined coordinate preflight target",
    )
    owner = _publication_owner(run, expected=expected_publication_owner)
    source_path = _canonical_path(
        source_subject_mask_path,
        prefix="subject_mask_runs/",
        label="raw subject-mask source",
    )
    if type(source_selector_eligible) is not bool:
        _fail("source_selector_eligible must be one exact boolean.")
    source = _source_metadata_context(
        root,
        source_path,
        selector_eligible=source_selector_eligible,
    )
    expected_source_name = source_path.split("/", 1)[1]
    if run.attrs.get("source_subject_mask_run") != expected_source_name:
        _fail(
            "Refined source_subject_mask_run does not name the exact selected raw run."
        )
    labels = _labels(mask_labels)
    if list(run.attrs.get("mask_labels", ())) != list(labels):
        _fail("Refined run mask_labels differ from the exact publication labels.")
    selection = _validate_exact_source_selection(source, run)
    if "masks_roi" not in run:
        _fail(
            "Future refined runs require physically present dense masks_roi; "
            "compact-only stores are historical compatibility surfaces."
        )
    created: list[str] = []
    targets: tuple[Any, ...] = ()
    snapshots: tuple[dict[str, Any], ...] = ()
    try:
        frames, made = _require_group(run, "coordinate_frames")
        if made:
            created.append(canonical_node_path(frames))
        continuous_node, made = _require_group(frames, "roi_local_continuous")
        if made:
            created.append(canonical_node_path(continuous_node))
        center_node, made = _require_group(frames, "roi_local_pixel_center")
        if made:
            created.append(canonical_node_path(center_node))
        edge_node, made = _require_group(
            frames,
            "roi_local_pixel_edge_half_open",
        )
        if made:
            created.append(canonical_node_path(edge_node))
        key_node = selection["instance_key"]
        time_node = selection["source_acquisition_frame_index"]
        placement = selection["source_crop_xywh"]
        targets, snapshots = _attrs_snapshot(
            run,
            key_node,
            time_node,
            placement,
            continuous_node,
            center_node,
            edge_node,
        )

        def authorize() -> Any:
            return _fresh_owned_ineligible_run(
                root,
                path,
                owner=owner,
                statuses=(RUN_STATUS_RUNNING,),
                label="Refined coordinate preflight mutation target",
            )

        authorize()
        identity = stamp_and_bind_row_identity_contract(
            run,
            key_node,
            contract=build_row_identity_contract(
                domain=OBSERVATION_INSTANCE_DOMAIN,
                values=_array(key_node, label="refined instance_key"),
            ),
        )
        temporal = stamp_source_row_temporal_authority(
            run,
            time_node,
            source_row_identity=identity,
            acquisition_frame=(
                source.source.crop_geometry.source_geometry.frame_evidence.acquisition_frame
            ),
        )
        label_authority = stamp_and_bind_persisted_coordinate_record(
            run,
            _label_record(labels),
            attr_name=REFINED_SUBJECT_MASK_COMPONENT_LABELS_ATTR,
        )
        assignment_keypoint_authority = stamp_and_bind_persisted_coordinate_record(
            run,
            _assignment_keypoint_record(
                source,
                run,
                assignment_keypoint_surfaces,
            ),
            attr_name=REFINED_SUBJECT_MASK_ASSIGNMENT_KEYPOINT_AUTHORITY_ATTR,
        )
        source_authority = stamp_and_bind_persisted_coordinate_record(
            run,
            _source_authority_record(
                source,
                run,
                labels,
                assignment_keypoints=assignment_keypoint_authority,
            ),
            attr_name=REFINED_SUBJECT_MASK_SOURCE_AUTHORITY_ATTR,
        )
        refinement_authority = stamp_and_bind_persisted_coordinate_record(
            run,
            _refinement_authority_record(
                run,
                source_authority=source_authority,
                assignment_keypoints=assignment_keypoint_authority,
                labels=labels,
            ),
            attr_name=REFINED_SUBJECT_MASK_REFINEMENT_AUTHORITY_ATTR,
        )
        width = int(source.continuous_frame.endpoint.width)
        height = int(source.continuous_frame.endpoint.height)
        if tuple(int(value) for value in run["masks_roi"].shape[2:]) != (
            height,
            width,
        ):
            _fail(
                "Refined masks_roi extent differs from the exact selected raw ROI extent."
            )
        rows_node = selection["source_crop_row_ids"]
        token = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]

        def frame_and_chain(
            frame_node: Any,
            *,
            convention: str,
            source_frame: BoundPixelFrameAuthority,
            target_camera: BoundPixelFrameAuthority,
            ownership_attr: str,
            authority_attr: str,
            transform_attr: str,
        ) -> tuple[BoundPixelFrameAuthority, BoundDirectedTransformChain]:
            authorize()
            ownership = stamp_crop_placement_ownership(
                placement,
                row_identity=identity,
                source_camera_frame=target_camera,
                attr_name=ownership_attr,
            )
            extent = _stamp_extent(
                frame_node,
                _extent_record(
                    width=width,
                    height=height,
                    convention=convention,
                    source_frame=source_frame,
                    source_rows_node=rows_node,
                ),
            )
            frame = stamp_roi_pixel_frame_authority(
                extent,
                frame_id=f"refined_subject_mask_roi_{convention}_{token}",
                pixel_convention=convention,
                crop_placement_ownership=ownership,
            )
            authority = stamp_crop_placement_transform_authority(
                placement,
                authority_id=(
                    f"refined_subject_mask_roi_{convention}_to_source_camera_{token}"
                ),
                source_frame=frame,
                target_frame=target_camera,
                attr_name=authority_attr,
            )
            link = stamp_directed_transform_v2(
                placement,
                transform_id=(
                    f"refined_subject_mask_roi_{convention}_to_source_camera_{token}"
                ),
                authority=authority,
                source_frame=frame,
                target_frame=target_camera,
                row_identity=identity,
                attr_name=transform_attr,
            )
            return frame, resolve_bound_directed_transform_chain((link,))

        continuous_frame, continuous_chain = frame_and_chain(
            continuous_node,
            convention="continuous",
            source_frame=source.continuous_frame,
            target_camera=source.continuous_chain.source_camera_frame_authority,
            ownership_attr=CROP_PLACEMENT_OWNERSHIP_ATTR,
            authority_attr=TRANSFORM_AUTHORITY_ATTR,
            transform_attr=DIRECTED_TRANSFORM_V2_ATTR,
        )
        center_frame, center_chain = frame_and_chain(
            center_node,
            convention="pixel_center",
            source_frame=source.pixel_center_frame,
            target_camera=source.pixel_center_chain.source_camera_frame_authority,
            ownership_attr=CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR,
            authority_attr=TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
            transform_attr=DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
        )
        edge_frame, edge_chain = frame_and_chain(
            edge_node,
            convention="pixel_edge_half_open",
            source_frame=source.pixel_edge_frame,
            target_camera=source.pixel_edge_chain.source_camera_frame_authority,
            ownership_attr=CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
            authority_attr=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
            transform_attr=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
        )
        record = _context_record(
            run_path=path,
            source=source,
            identity=identity,
            temporal=temporal,
            labels=label_authority,
            assignment_keypoints=assignment_keypoint_authority,
            source_authority=source_authority,
            refinement_authority=refinement_authority,
            continuous_frame=continuous_frame,
            continuous_chain=continuous_chain,
            center_frame=center_frame,
            center_chain=center_chain,
            edge_frame=edge_frame,
            edge_chain=edge_chain,
            selection=selection,
            owner=owner,
            source_selector_eligible=source_selector_eligible,
        )
        authorize()
        context_bound = stamp_and_bind_persisted_coordinate_record(
            run,
            record,
            attr_name=REFINED_SUBJECT_MASK_COORDINATE_CONTEXT_ATTR,
        )
        return BoundRefinedSubjectMaskCoordinateContext(
            source=source,
            row_identity=identity,
            temporal_authority=temporal,
            component_labels=label_authority,
            assignment_keypoint_authority=assignment_keypoint_authority,
            assignment_keypoint_surfaces=assignment_keypoint_surfaces,
            source_authority=source_authority,
            refinement_authority=refinement_authority,
            labels=labels,
            continuous_frame=continuous_frame,
            continuous_chain=continuous_chain,
            pixel_center_frame=center_frame,
            pixel_center_chain=center_chain,
            pixel_edge_frame=edge_frame,
            pixel_edge_chain=edge_chain,
            context_record=context_bound,
            run_path=path,
            completion_status=RUN_STATUS_RUNNING,
            selector_eligible=False,
            publication_owner=owner,
            _root=root,
            _run_group=run,
            _verification_seal=_BOUND_CONTEXT_SEAL,
        )
    except BaseException as exc:
        failures: list[str] = []
        try:
            _restore_attrs(
                root,
                targets,
                snapshots,
                run_path=path,
                owner=owner,
            )
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            failures.append(f"attrs: {rollback_exc}")
        try:
            _delete_created(root, created, run_path=path, owner=owner)
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            failures.append(f"nodes: {rollback_exc}")
        if failures:
            raise RefinedSubjectMaskCoordinatePublicationError(
                "Refined context preparation failed and rollback was incomplete: "
                f"{failures!r}."
            ) from exc
        raise


def _load_refined_subject_mask_coordinate_context(
    root: Any,
    run_path: str,
    *,
    require_complete: bool,
    require_activation_receipt: bool | None = None,
    expected_selector_eligible: bool,
    expected_publication_owner: str | None = None,
) -> BoundRefinedSubjectMaskCoordinateContext:
    path = _canonical_path(
        run_path,
        prefix="refined_subject_masks_runs/",
        label="refined subject-mask rowset",
    )
    run = _node(root, path, label="refined subject-mask rowset")
    owner = _publication_owner(run, expected=expected_publication_owner)
    status = RUN_STATUS_COMPLETE if require_complete else RUN_STATUS_RUNNING
    _require_run_status(
        run,
        status=status,
        selector_eligible=expected_selector_eligible,
        label="Canonical refined subject-mask rowset",
    )
    if require_complete and run.attrs.get("coordinate_contract") != "canonical_v2":
        _fail("Complete refined coordinate contexts require canonical_v2 publication.")
    context = bind_persisted_coordinate_record(
        run,
        attr_name=REFINED_SUBJECT_MASK_COORDINATE_CONTEXT_ATTR,
    )
    source_path = context.record.get("source_subject_mask_path")
    source_selector_eligible = context.record.get("source_selector_eligible", True)
    if type(source_selector_eligible) is not bool:
        _fail("Persisted refined raw-source eligibility policy is malformed.")
    source = _source_metadata_context(
        root,
        source_path,
        selector_eligible=source_selector_eligible,
    )
    if run.attrs.get("source_subject_mask_run") != source_path.split("/", 1)[1]:
        _fail("Refined source run attr changed after coordinate publication.")
    selection = _validate_exact_source_selection(source, run)
    labels_record = bind_persisted_coordinate_record(
        run,
        attr_name=REFINED_SUBJECT_MASK_COMPONENT_LABELS_ATTR,
    )
    labels = _labels(labels_record.record.get("labels", ()))
    if labels_record.record != _label_record(labels) or list(
        run.attrs.get("mask_labels", ())
    ) != list(labels):
        _fail("Persisted refined component labels differ from exact authority.")
    assignment_keypoint_authority = bind_persisted_coordinate_record(
        run,
        attr_name=REFINED_SUBJECT_MASK_ASSIGNMENT_KEYPOINT_AUTHORITY_ATTR,
    )
    assignment_status = assignment_keypoint_authority.record.get("status")
    assignment_keypoint_surfaces: BoundKeypointCoordinateSurfaces | None
    if assignment_status == "used":
        keypoint_path = assignment_keypoint_authority.record.get("keypoint_run_path")
        if not isinstance(keypoint_path, str):
            _fail("Persisted assignment keypoint authority lacks an exact run path.")
        try:
            assignment_keypoint_surfaces = load_persisted_keypoint_coordinate_surfaces(
                root,
                keypoint_path,
            )
        except Exception as exc:
            raise RefinedSubjectMaskCoordinatePublicationError(
                f"Persisted assignment keypoint dependency is stale or invalid: {exc}"
            ) from exc
    elif assignment_status == "not_used":
        assignment_keypoint_surfaces = None
    else:
        _fail("Persisted assignment keypoint authority has unsupported status.")
    if assignment_keypoint_authority.record != _assignment_keypoint_record(
        source,
        run,
        assignment_keypoint_surfaces,
    ):
        _fail(
            "Persisted assignment keypoint authority differs from exact live evidence."
        )
    source_authority = bind_persisted_coordinate_record(
        run,
        attr_name=REFINED_SUBJECT_MASK_SOURCE_AUTHORITY_ATTR,
    )
    if source_authority.record != _source_authority_record(
        source,
        run,
        labels,
        assignment_keypoints=assignment_keypoint_authority,
    ):
        _fail("Persisted refined source authority differs from live raw metadata.")
    refinement_authority = bind_persisted_coordinate_record(
        run,
        attr_name=REFINED_SUBJECT_MASK_REFINEMENT_AUTHORITY_ATTR,
    )
    if refinement_authority.record != _refinement_authority_record(
        run,
        source_authority=source_authority,
        assignment_keypoints=assignment_keypoint_authority,
        labels=labels,
    ):
        _fail("Persisted refinement authority differs from live provenance.")
    if require_activation_receipt is None:
        require_activation_receipt = require_complete
    if require_activation_receipt and not require_complete:
        _fail("Activation receipts are valid only for completed refined runs.")
    if require_activation_receipt:
        activation_receipt = bind_persisted_coordinate_record(
            run,
            attr_name=REFINED_SUBJECT_MASK_ACTIVATION_RECEIPT_ATTR,
        )
        if activation_receipt.record != _activation_receipt_record(
            run,
            refinement_authority=refinement_authority,
        ):
            _fail(
                "Persisted refined activation receipt differs from live completion provenance."
            )
    identity = load_bound_row_identity_contract(
        run,
        selection["instance_key"],
    )
    temporal = load_bound_source_row_temporal_authority(
        run,
        selection["source_acquisition_frame_index"],
        source_row_identity=identity,
        acquisition_frame=(
            source.source.crop_geometry.source_geometry.frame_evidence.acquisition_frame
        ),
    )
    width = int(source.continuous_frame.endpoint.width)
    height = int(source.continuous_frame.endpoint.height)
    masks = _child(run, "masks_roi", label="authoritative refined dense masks")
    if tuple(int(value) for value in masks.shape[2:]) != (height, width):
        _fail("Refined dense mask extent changed from the exact raw ROI extent.")
    rows_node = selection["source_crop_row_ids"]
    placement = selection["source_crop_xywh"]
    historical_padded_source = (
        "coordinate_successor_historical_crop_adapter"
        in getattr(source._run_group, "attrs", {})
    )
    continuous_ownership_attr = (
        CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR
        if historical_padded_source
        else CROP_PLACEMENT_OWNERSHIP_ATTR
    )
    center_ownership_attr = (
        CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR
        if historical_padded_source
        else CROP_PLACEMENT_PIXEL_CENTER_OWNERSHIP_ATTR
    )
    edge_ownership_attr = (
        CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR
        if historical_padded_source
        else CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR
    )

    def load_frame_and_chain(
        frame_name: str,
        *,
        convention: str,
        source_frame: BoundPixelFrameAuthority,
        target_camera: BoundPixelFrameAuthority,
        ownership_attr: str,
        authority_attr: str,
        transform_attr: str,
    ) -> tuple[BoundPixelFrameAuthority, BoundDirectedTransformChain]:
        ownership = load_crop_placement_ownership(
            placement,
            row_identity=identity,
            source_camera_frame=target_camera,
            attr_name=ownership_attr,
        )
        frame_node = _node(
            root,
            f"{path}/coordinate_frames/{frame_name}",
            label=f"refined {convention} ROI frame",
        )
        extent = bind_persisted_record_reference_extent(
            frame_node,
            record_attr=REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR,
            digest_attr=f"{REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR}_sha256",
            width_field="width",
            height_field="height",
            units_field="units",
        )
        expected_extent = _extent_record(
            width=width,
            height=height,
            convention=convention,
            source_frame=source_frame,
            source_rows_node=rows_node,
        )
        expected_bound = {
            **expected_extent,
            "bound_record_attr": REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR,
            "bound_digest_attr": f"{REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR}_sha256",
            "bound_width_field": "width",
            "bound_height_field": "height",
            "bound_units_field": "units",
        }
        if extent.authority_record != expected_bound:
            _fail("Persisted refined ROI extent differs from exact raw extent.")
        frame = load_roi_pixel_frame_authority(
            frame_node,
            reference_extent=extent,
            crop_placement_ownership=ownership,
        )
        authority = load_bound_transform_authority(
            placement,
            payload_node=placement,
            source_frame=frame,
            target_frame=target_camera,
            row_identity=identity,
            attr_name=authority_attr,
        )
        link = load_bound_directed_transform_v2(
            placement,
            authority=authority,
            source_frame=frame,
            target_frame=target_camera,
            row_identity=identity,
            attr_name=transform_attr,
        )
        return frame, resolve_bound_directed_transform_chain((link,))

    continuous_frame, continuous_chain = load_frame_and_chain(
        "roi_local_continuous",
        convention="continuous",
        source_frame=source.continuous_frame,
        target_camera=source.continuous_chain.source_camera_frame_authority,
        ownership_attr=continuous_ownership_attr,
        authority_attr=TRANSFORM_AUTHORITY_ATTR,
        transform_attr=DIRECTED_TRANSFORM_V2_ATTR,
    )
    center_frame, center_chain = load_frame_and_chain(
        "roi_local_pixel_center",
        convention="pixel_center",
        source_frame=source.pixel_center_frame,
        target_camera=source.pixel_center_chain.source_camera_frame_authority,
        ownership_attr=center_ownership_attr,
        authority_attr=TRANSFORM_AUTHORITY_PIXEL_CENTER_ATTR,
        transform_attr=DIRECTED_TRANSFORM_V2_PIXEL_CENTER_ATTR,
    )
    edge_frame, edge_chain = load_frame_and_chain(
        "roi_local_pixel_edge_half_open",
        convention="pixel_edge_half_open",
        source_frame=source.pixel_edge_frame,
        target_camera=source.pixel_edge_chain.source_camera_frame_authority,
        ownership_attr=edge_ownership_attr,
        authority_attr=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
        transform_attr=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    )
    expected_context = _context_record(
        run_path=path,
        source=source,
        identity=identity,
        temporal=temporal,
        labels=labels_record,
        assignment_keypoints=assignment_keypoint_authority,
        source_authority=source_authority,
        refinement_authority=refinement_authority,
        continuous_frame=continuous_frame,
        continuous_chain=continuous_chain,
        center_frame=center_frame,
        center_chain=center_chain,
        edge_frame=edge_frame,
        edge_chain=edge_chain,
        selection=selection,
        owner=owner,
        source_selector_eligible=source_selector_eligible,
    )
    if context.record != expected_context:
        _fail("Persisted refined coordinate context differs from live exact evidence.")
    return BoundRefinedSubjectMaskCoordinateContext(
        source=source,
        row_identity=identity,
        temporal_authority=temporal,
        component_labels=labels_record,
        assignment_keypoint_authority=assignment_keypoint_authority,
        assignment_keypoint_surfaces=assignment_keypoint_surfaces,
        source_authority=source_authority,
        refinement_authority=refinement_authority,
        labels=labels,
        continuous_frame=continuous_frame,
        continuous_chain=continuous_chain,
        pixel_center_frame=center_frame,
        pixel_center_chain=center_chain,
        pixel_edge_frame=edge_frame,
        pixel_edge_chain=edge_chain,
        context_record=context,
        run_path=path,
        completion_status=status,
        selector_eligible=expected_selector_eligible,
        publication_owner=owner,
        _root=root,
        _run_group=run,
        _verification_seal=_BOUND_CONTEXT_SEAL,
    )


def _digest_values(
    digest: Any,
    node: Any,
    values: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype.hasobject or result.dtype != np.dtype(node.dtype):
        _fail(f"{label} changed dtype during its payload scan.")
    contiguous = np.ascontiguousarray(result)
    digest.update(contiguous.tobytes(order="C"))
    return contiguous


def _finish_payload(digest: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {**copy.deepcopy(dict(metadata)), "array_values_sha256": digest.hexdigest()}


def _require_shape_dtype(
    node: Any,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    label: str,
) -> None:
    if (
        tuple(int(item) for item in node.shape) != shape
        or np.dtype(node.dtype) != dtype
    ):
        _fail(
            f"{label} must have exact shape {shape!r} and dtype {dtype.str!r}; "
            f"got {tuple(node.shape)!r}/{np.dtype(node.dtype).str!r}."
        )


def _derive_mask_metrics(masks: np.ndarray) -> dict[str, np.ndarray]:
    binary = np.asarray(masks, dtype=np.uint8) != 0
    rows, components, height, width = map(int, binary.shape)
    flat = binary.reshape(rows * components, height, width)
    area = (
        flat.reshape(rows * components, -1)
        .sum(axis=1, dtype=np.int64)
        .astype(np.float32)
    )
    valid = area > 0.0
    centroid = np.zeros((rows * components, 2), dtype=np.float32)
    bbox = np.zeros((rows * components, 4), dtype=np.float32)
    if bool(np.any(valid)):
        y_counts = flat.sum(axis=2, dtype=np.float32)
        x_counts = flat.sum(axis=1, dtype=np.float32)
        x_coords = np.arange(width, dtype=np.float32)
        y_coords = np.arange(height, dtype=np.float32)
        denominator = np.maximum(area, 1.0).astype(np.float32, copy=False)
        centroid[:, 0] = np.asarray(x_counts @ x_coords, dtype=np.float32) / denominator
        centroid[:, 1] = np.asarray(y_counts @ y_coords, dtype=np.float32) / denominator
        row_has = flat.any(axis=2)
        col_has = flat.any(axis=1)
        y_index = np.arange(height, dtype=np.int32).reshape(1, height)
        x_index = np.arange(width, dtype=np.int32).reshape(1, width)
        y_min = np.where(row_has, y_index, height).min(axis=1)
        y_max_exclusive = np.where(row_has, y_index + 1, 0).max(axis=1)
        x_min = np.where(col_has, x_index, width).min(axis=1)
        x_max_exclusive = np.where(col_has, x_index + 1, 0).max(axis=1)
        bbox[:] = np.stack(
            (x_min, y_min, x_max_exclusive, y_max_exclusive),
            axis=1,
        ).astype(np.float32, copy=False)
        centroid[~valid] = 0.0
        bbox[~valid] = 0.0
    return {
        "mask_present": valid.reshape(rows, components),
        "area_px": area.reshape(rows, components),
        "centroid_xy": centroid.reshape(rows, components, 2),
        "centroid_valid": valid.reshape(rows, components),
        "bbox_xyxy": bbox.reshape(rows, components, 4),
        "bbox_valid": valid.reshape(rows, components),
    }


def _scan_required_surfaces(
    context: BoundRefinedSubjectMaskCoordinateContext,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    run = context._run_group
    masks = _child(run, "masks_roi", label="authoritative refined dense masks")
    metrics = _child(run, "metrics", label="refined run metrics")
    available = _child(run, "available_channels", label="available channels")
    _require_exact_members(
        metrics,
        (
            "mask_present",
            "area_px",
            "centroid_xy",
            "centroid_valid",
            "bbox_xyxy",
            "bbox_valid",
        ),
        label="refined run metrics",
    )
    for attr_name in (
        "derived_mask_caches_stale",
        "metrics_stale",
        "contours_stale",
    ):
        if run.attrs.get(attr_name) is not False:
            _fail(
                f"Future refined publication requires explicit fresh {attr_name}=False."
            )
    for cache_name in ("mask_bitpacked", "mask_rle"):
        if (
            _optional_child(run, cache_name) is not None
            and run.attrs.get(f"{cache_name}_stale") is not False
        ):
            _fail(f"Published {cache_name} must explicitly be a fresh derived cache.")
    nodes = {
        "masks_roi": masks,
        "mask_present": _child(metrics, "mask_present", label="mask_present"),
        "area_px": _child(metrics, "area_px", label="area_px"),
        "centroid_xy": _child(metrics, "centroid_xy", label="centroid_xy"),
        "centroid_valid": _child(metrics, "centroid_valid", label="centroid_valid"),
        "bbox_xyxy": _child(metrics, "bbox_xyxy", label="bbox_xyxy"),
        "bbox_valid": _child(metrics, "bbox_valid", label="bbox_valid"),
        "available_channels": available,
    }
    shape = tuple(int(value) for value in masks.shape)
    if len(shape) != 4 or any(value <= 0 for value in shape):
        _fail("Refined masks_roi must have nonempty shape (N,C,H,W).")
    rows, components, height, width = shape
    if rows != context.row_identity.leading_dimension:
        _fail(
            "Refined masks_roi leading dimension differs from instance_key authority."
        )
    if components != len(context.labels):
        _fail("Refined masks_roi channel count differs from ordered labels.")
    if (height, width) != (
        int(context.pixel_center_frame.endpoint.height),
        int(context.pixel_center_frame.endpoint.width),
    ):
        _fail("Refined masks_roi extent differs from the bound ROI frame.")
    _require_shape_dtype(
        masks,
        shape=shape,
        dtype=np.dtype("uint8"),
        label="masks_roi",
    )
    specs = {
        "mask_present": ((rows, components), np.dtype("bool")),
        "area_px": ((rows, components), np.dtype("float32")),
        "centroid_xy": ((rows, components, 2), np.dtype("float32")),
        "centroid_valid": ((rows, components), np.dtype("bool")),
        "bbox_xyxy": ((rows, components, 4), np.dtype("float32")),
        "bbox_valid": ((rows, components), np.dtype("bool")),
        "available_channels": ((components,), np.dtype("bool")),
    }
    for name, (expected_shape, dtype) in specs.items():
        _require_shape_dtype(
            nodes[name],
            shape=expected_shape,
            dtype=dtype,
            label=name,
        )
    if run.attrs.get("bbox_xyxy_convention") != "pixel_edge_half_open":
        _fail("Refined bbox_xyxy requires explicit pixel_edge_half_open convention.")
    if (
        run.attrs.get("bbox_xyxy_derivation")
        != "foreground_half_open_pixel_edges_xyxy_v1"
    ):
        _fail("Refined bbox_xyxy lacks the exact half-open derivation label.")

    states = {name: _payload_digest_state(node) for name, node in nodes.items()}
    available_values = _array(available, label="available_channels")
    _digest_values(
        states["available_channels"][0],
        available,
        available_values,
        label="available_channels",
    )
    if not bool(np.all(available_values)):
        _fail(
            "Future refined outputs must not publish unavailable declared components."
        )

    chunk_rows = _payload_row_chunk(masks)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        mask_values = _digest_values(
            states["masks_roi"][0],
            masks,
            np.asarray(masks[start:stop]),
            label="masks_roi",
        )
        if not bool(np.all((mask_values == 0) | (mask_values == 1))):
            _fail("Authoritative refined masks_roi must contain only uint8 0/1 values.")
        derived = _derive_mask_metrics(mask_values)
        for name in (
            "mask_present",
            "area_px",
            "centroid_xy",
            "centroid_valid",
            "bbox_xyxy",
            "bbox_valid",
        ):
            node = nodes[name]
            actual = _digest_values(
                states[name][0],
                node,
                np.asarray(node[start:stop]),
                label=name,
            )
            expected = derived[name]
            if name == "centroid_xy":
                if not np.allclose(actual, expected, rtol=0.0, atol=2e-5):
                    _fail("Refined centroid_xy differs from authoritative dense masks.")
            elif not np.array_equal(actual, expected):
                _fail(
                    f"Refined {name} differs from authoritative dense masks or "
                    "uses the wrong bbox convention."
                )
    payloads = {
        name: _finish_payload(digest, metadata)
        for name, (digest, metadata) in states.items()
    }
    # Every required surface has now been completely read, semantically
    # checked, and hashed.  Seed the call-scoped cache only at this point so
    # later inventory/manifest construction cannot repeat those payload scans.
    for name, payload in payloads.items():
        _payload_cache_store(nodes[name], payload)
    return nodes, payloads


def _component_group(run: Any, component: str) -> Any | None:
    components = _optional_child(run, "components")
    if components is None:
        return None
    return _optional_child(components, component)


def _validate_optional_geometry(
    context: BoundRefinedSubjectMaskCoordinateContext,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Return coordinate, ragged, and scalar optional surface specifications."""

    rows = context.row_identity.leading_dimension
    width = int(context.pixel_center_frame.endpoint.width)
    height = int(context.pixel_center_frame.endpoint.height)
    canonical: dict[str, dict[str, Any]] = {}
    ragged: dict[str, dict[str, Any]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    for component in context.labels:
        group = _component_group(context._run_group, component)
        if group is None:
            continue
        geometry = _optional_child(group, "geometry")
        if geometry is not None:
            _require_exact_members(
                geometry,
                ("ellipse_params", "ellipse_success"),
                label=f"{component} geometry",
            )
            ellipse = _child(geometry, "ellipse_params", label=f"{component} ellipse")
            success = _child(
                geometry, "ellipse_success", label=f"{component} ellipse validity"
            )
            _require_shape_dtype(
                ellipse,
                shape=(rows, 5),
                dtype=np.dtype("float32"),
                label=f"{component} ellipse_params",
            )
            _require_shape_dtype(
                success,
                shape=(rows,),
                dtype=np.dtype("bool"),
                label=f"{component} ellipse_success",
            )
            values = _array(ellipse, label=f"{component} ellipse_params")
            valid = _array(success, label=f"{component} ellipse_success")
            fit_diagnostics = {
                "policy": "diagnostic_only_does_not_override_opencv_fit_success_v1",
                "algorithm_success_count": int(np.count_nonzero(valid)),
                "center_outside_roi_count": 0,
                "axis_larger_than_roi_extent_count": 0,
            }
            if bool(np.any(valid)):
                selected = values[valid]
                if (
                    not bool(np.isfinite(selected).all())
                    or bool(np.any(selected[:, 2:4] <= 0.0))
                    or bool(np.any(selected[:, 2] < selected[:, 3]))
                    or bool(np.any(selected[:, 4] < 0.0))
                    or bool(np.any(selected[:, 4] >= 180.0))
                ):
                    _fail(
                        f"{component} valid ellipse geometry violates the exact "
                        "normalized OpenCV fit-result contract."
                    )
                center_outside = (
                    (selected[:, 0] < 0.0)
                    | (selected[:, 0] >= width)
                    | (selected[:, 1] < 0.0)
                    | (selected[:, 1] >= height)
                )
                axis_larger_than_extent = (selected[:, 2] > max(width, height)) | (
                    selected[:, 3] > max(width, height)
                )
                fit_diagnostics["center_outside_roi_count"] = int(
                    np.count_nonzero(center_outside)
                )
                fit_diagnostics["axis_larger_than_roi_extent_count"] = int(
                    np.count_nonzero(axis_larger_than_extent)
                )
            if bool(np.any(~valid)) and not bool(np.isnan(values[~valid]).all()):
                _fail(
                    f"{component} invalid ellipse geometry must use an all-NaN sentinel."
                )
            key = f"components/{component}/geometry/ellipse_params"
            canonical[key] = {
                "node": ellipse,
                "validity_node": success,
                "geometry_type": "ellipse_cxcy_wh_angle",
                "components": ("center_x", "center_y", "width", "height", "angle"),
                "component_units": ("px", "px", "px", "px", "deg"),
                "pixel_convention": "continuous",
                "frame": context.continuous_frame,
                "chain": context.continuous_chain,
                "overlay": CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
                "component": component,
                "container": geometry,
                "fit_result_diagnostics": fit_diagnostics,
                "invalid_value_policy": "all_nan_geometry_is_invalid_sentinel_not_coordinate",
            }
        sampled = _optional_child(group, "sampled_contours")
        if sampled is not None:
            _require_exact_members(
                sampled,
                ("points_xy", "source_point_count", "valid"),
                label=f"{component} sampled_contours",
            )
            points = _child(sampled, "points_xy", label=f"{component} sampled contour")
            valid_node = _child(
                sampled, "valid", label=f"{component} sampled contour validity"
            )
            source_point_count = _child(
                sampled,
                "source_point_count",
                label=f"{component} sampled contour source count",
            )
            shape = tuple(int(value) for value in points.shape)
            if len(shape) != 3 or shape[0] != rows or shape[1] <= 0 or shape[2] != 2:
                _fail(f"{component} sampled contour must have shape (N,K,2).")
            _require_shape_dtype(
                points,
                shape=shape,
                dtype=np.dtype("float32"),
                label=f"{component} sampled contour",
            )
            _require_shape_dtype(
                valid_node,
                shape=(rows,),
                dtype=np.dtype("bool"),
                label=f"{component} sampled contour validity",
            )
            _require_shape_dtype(
                source_point_count,
                shape=(rows,),
                dtype=np.dtype("int32"),
                label=f"{component} sampled contour source_point_count",
            )
            values = _array(points, label=f"{component} sampled contour")
            valid = _array(valid_node, label=f"{component} sampled contour validity")
            source_counts = _array(
                source_point_count,
                label=f"{component} sampled contour source_point_count",
            )
            if bool(np.any(source_counts < 0)):
                _fail(f"{component} sampled contour source counts must be nonnegative.")
            if bool(np.any(valid)):
                selected = values[valid]
                if (
                    not bool(np.isfinite(selected).all())
                    or bool(np.any(selected[..., 0] < 0.0))
                    or bool(np.any(selected[..., 0] >= width))
                    or bool(np.any(selected[..., 1] < 0.0))
                    or bool(np.any(selected[..., 1] >= height))
                ):
                    _fail(f"{component} sampled contour leaves the exact ROI extent.")
            if bool(np.any(~valid)) and not bool(np.isnan(values[~valid]).all()):
                _fail(
                    f"{component} invalid sampled contours must use an all-NaN sentinel."
                )
            key = f"components/{component}/sampled_contours/points_xy"
            canonical[key] = {
                "node": points,
                "validity_node": valid_node,
                "geometry_type": "points_xy",
                "components": ("x", "y"),
                "component_units": ("px", "px"),
                "pixel_convention": "pixel_center",
                "frame": context.pixel_center_frame,
                "chain": context.pixel_center_chain,
                "overlay": CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
                "component": component,
                "container": sampled,
                "companion_nodes": {"source_point_count": source_point_count},
                "invalid_value_policy": "all_nan_geometry_is_invalid_sentinel_not_coordinate",
            }
        contours = _optional_child(group, "contours")
        if contours is not None:
            _require_exact_members(
                contours,
                ("len", "points_xy", "ptr"),
                label=f"{component} contours",
            )
            points = _child(contours, "points_xy", label=f"{component} packed contour")
            ptr = _child(contours, "ptr", label=f"{component} contour ptr")
            length = _child(contours, "len", label=f"{component} contour len")
            point_shape = tuple(int(value) for value in points.shape)
            if len(point_shape) != 2 or point_shape[1] != 2 or point_shape[0] <= 0:
                _fail(f"{component} packed contour points_xy must have shape (P,2).")
            _require_shape_dtype(
                points,
                shape=point_shape,
                dtype=np.dtype("float32"),
                label=f"{component} contour points",
            )
            _require_shape_dtype(
                ptr,
                shape=(rows,),
                dtype=np.dtype("int64"),
                label=f"{component} contour ptr",
            )
            _require_shape_dtype(
                length,
                shape=(rows,),
                dtype=np.dtype("int32"),
                label=f"{component} contour len",
            )
            points_values = _array(points, label=f"{component} contour points")
            ptr_values = _array(ptr, label=f"{component} contour ptr").astype(
                np.int64, copy=False
            )
            len_values = _array(length, label=f"{component} contour len").astype(
                np.int64, copy=False
            )
            invalid = len_values == 0
            if (
                bool(np.any(len_values < 0))
                or bool(np.any(ptr_values[invalid] != -1))
                or bool(np.any(ptr_values[~invalid] < 0))
                or bool(
                    np.any(ptr_values[~invalid] + len_values[~invalid] > point_shape[0])
                )
            ):
                _fail(f"{component} packed contour row mapping is invalid.")
            expected_offset = 0
            for offset, count in zip(
                ptr_values[~invalid], len_values[~invalid], strict=True
            ):
                if int(offset) != expected_offset:
                    _fail(
                        f"{component} packed contour row segments are not canonical/contiguous."
                    )
                expected_offset += int(count)
            placeholder = bool(contours.attrs.get("points_placeholder_when_empty"))
            if expected_offset == 0:
                if not placeholder or point_shape[0] != 1:
                    _fail(
                        f"{component} empty contour store lacks its exact placeholder policy."
                    )
            elif expected_offset != point_shape[0]:
                _fail(
                    f"{component} contour points are not fully owned by observation rows."
                )
            if expected_offset > 0:
                owned = points_values[:expected_offset]
                if (
                    not bool(np.isfinite(owned).all())
                    or bool(np.any(owned[:, 0] < 0.0))
                    or bool(np.any(owned[:, 0] >= width))
                    or bool(np.any(owned[:, 1] < 0.0))
                    or bool(np.any(owned[:, 1] >= height))
                ):
                    _fail(f"{component} packed contour leaves the exact ROI extent.")
            key = f"components/{component}/contours/points_xy"
            ragged[key] = {
                "node": points,
                "ptr_node": ptr,
                "len_node": length,
                "component": component,
                "point_count": expected_offset,
                "container": contours,
            }
    relations = _optional_child(context._run_group, "relations")
    eye_pair = _optional_child(relations, "eye_pair") if relations is not None else None
    relation_metrics = (
        _optional_child(eye_pair, "metrics") if eye_pair is not None else None
    )
    if relation_metrics is not None:
        if eye_pair is None:
            _fail("Eye-pair relation metrics lack their exact relation container.")
        _require_exact_members(
            eye_pair,
            ("metrics",),
            label="eye-pair relation",
        )
        _require_exact_members(
            relation_metrics,
            ("separation_px", "separation_valid"),
            label="eye-pair relation metrics",
        )
        separation = _child(
            relation_metrics,
            "separation_px",
            label="eye-pair separation",
        )
        valid_node = _child(
            relation_metrics,
            "separation_valid",
            label="eye-pair separation validity",
        )
        _require_shape_dtype(
            separation,
            shape=(rows,),
            dtype=np.dtype("float32"),
            label="eye-pair separation_px",
        )
        _require_shape_dtype(
            valid_node,
            shape=(rows,),
            dtype=np.dtype("bool"),
            label="eye-pair separation_valid",
        )
        values = _array(separation, label="eye-pair separation_px")
        valid = _array(valid_node, label="eye-pair separation_valid")
        expected_components = ("eye_left", "eye_right")
        attrs = _exact_json_attrs(
            relation_metrics,
            label="eye-pair relation metrics",
        )
        if (
            attrs.get("relation_components") != list(expected_components)
            or attrs.get("relation_method") != "ellipse_centroid_distance"
        ):
            _fail(
                "Eye-pair separation lacks direction-independent anatomical component "
                "identity and exact ellipse-centroid method metadata."
            )
        ellipse_paths = tuple(
            f"components/{component}/geometry/ellipse_params"
            for component in expected_components
        )
        if any(path not in canonical for path in ellipse_paths):
            _fail(
                "Eye-pair separation requires sealed eye_left and eye_right ellipse geometry."
            )
        ellipse_values = tuple(
            _array(canonical[path]["node"], label=f"{path} relation source")
            for path in ellipse_paths
        )
        ellipse_valid = tuple(
            _array(canonical[path]["validity_node"], label=f"{path} validity")
            for path in ellipse_paths
        )
        expected_valid = np.logical_and(ellipse_valid[0], ellipse_valid[1])
        if not np.array_equal(valid, expected_valid):
            _fail("Eye-pair separation_valid differs from exact eye ellipse validity.")
        expected_values = np.linalg.norm(
            ellipse_values[0][:, :2] - ellipse_values[1][:, :2],
            axis=1,
        ).astype(np.float32, copy=False)
        if bool(np.any(valid)) and not np.allclose(
            values[valid],
            expected_values[valid],
            rtol=1e-6,
            atol=1e-5,
        ):
            _fail(
                "Eye-pair separation differs from the exact selected eye ellipse centers."
            )
        if bool(np.any(~valid)) and not bool(np.isnan(values[~valid]).all()):
            _fail("Invalid eye-pair separations must use a NaN sentinel.")
        measurements["relations/eye_pair/metrics/separation_px"] = {
            "node": separation,
            "validity_node": valid_node,
            "quantity": "eye_pair_separation",
            "units": "px",
            "operation": "euclidean_distance_between_refined_eye_ellipse_centers_v1",
            "axes": ("observation",),
            "coordinate_input_paths": ellipse_paths,
            "measurement_input_paths": (),
            "row_axis_name": "observation",
            "collection_axis": False,
            "selected_collection_members": expected_components,
            "semantic_kind": "distance",
            "validity_policy": "nan_when_either_eye_ellipse_invalid_v1",
        }
    return canonical, ragged, measurements


def _required_geometry_specs(
    context: BoundRefinedSubjectMaskCoordinateContext,
    nodes: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "masks_roi": {
            "node": nodes["masks_roi"],
            "validity_node": None,
            "geometry_type": "raster_yx",
            "components": ("y", "x"),
            "component_units": ("px", "px"),
            "pixel_convention": "pixel_center",
            "frame": context.pixel_center_frame,
            "chain": context.pixel_center_chain,
            "overlay": CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            "component": None,
            "collection_axis": True,
        },
        "metrics/centroid_xy": {
            "node": nodes["centroid_xy"],
            "validity_node": nodes["centroid_valid"],
            "geometry_type": "point_xy",
            "components": ("x", "y"),
            "component_units": ("px", "px"),
            "pixel_convention": "continuous",
            "frame": context.continuous_frame,
            "chain": context.continuous_chain,
            "overlay": CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            "component": None,
            "collection_axis": True,
        },
        "metrics/bbox_xyxy": {
            "node": nodes["bbox_xyxy"],
            "validity_node": nodes["bbox_valid"],
            "geometry_type": "bbox_xyxy",
            "components": ("x_min", "y_min", "x_max", "y_max"),
            "component_units": ("px", "px", "px", "px"),
            "pixel_convention": "pixel_edge_half_open",
            "frame": context.pixel_edge_frame,
            "chain": context.pixel_edge_chain,
            "overlay": CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            "component": None,
            "collection_axis": True,
        },
    }


def _scalar_spec(
    node: Any,
    *,
    quantity: str,
    units: str,
    operation: str,
    axes: Sequence[str],
    coordinate_input_paths: Sequence[str] = (),
    measurement_input_paths: Sequence[str] = (),
    row_axis_name: str | None = None,
    collection_axis: bool = False,
    selected_collection_members: Sequence[str] = (),
    semantic_kind: str,
    validity_node: Any | None = None,
    validity_policy: str | None = None,
    record_inputs: Sequence[BoundCoordinateRecord] = (),
) -> dict[str, Any]:
    return {
        "node": node,
        "quantity": quantity,
        "units": units,
        "operation": operation,
        "axes": tuple(axes),
        "coordinate_input_paths": tuple(coordinate_input_paths),
        "measurement_input_paths": tuple(measurement_input_paths),
        "row_axis_name": row_axis_name,
        "collection_axis": bool(collection_axis),
        "selected_collection_members": tuple(selected_collection_members),
        "semantic_kind": semantic_kind,
        "validity_node": validity_node,
        "validity_policy": validity_policy,
        "record_inputs": tuple(record_inputs),
    }


def _measurement_specs(
    context: BoundRefinedSubjectMaskCoordinateContext,
    required: Mapping[str, Any],
    coordinate_specs: Mapping[str, Mapping[str, Any]],
    ragged_specs: Mapping[str, Mapping[str, Any]],
    relation_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate and classify every retained scientific scalar leaf."""

    rows = context.row_identity.leading_dimension
    result: dict[str, dict[str, Any]] = {
        "metrics/mask_present": _scalar_spec(
            required["mask_present"],
            quantity="mask_presence",
            units="1",
            operation="foreground_pixel_count_greater_than_zero_v1",
            axes=("observation", "subject_component"),
            coordinate_input_paths=("masks_roi",),
            row_axis_name="observation",
            collection_axis=True,
            semantic_kind="validity",
        ),
        "metrics/area_px": _scalar_spec(
            required["area_px"],
            quantity="foreground_area",
            units="px^2",
            operation="foreground_pixel_count_from_authoritative_masks_roi_v1",
            axes=("observation", "subject_component"),
            coordinate_input_paths=("masks_roi",),
            row_axis_name="observation",
            collection_axis=True,
            semantic_kind="area",
            validity_node=required["mask_present"],
            validity_policy="zero_when_mask_absent_v1",
        ),
        "metrics/centroid_valid": _scalar_spec(
            required["centroid_valid"],
            quantity="centroid_validity",
            units="1",
            operation="foreground_pixel_count_greater_than_zero_v1",
            axes=("observation", "subject_component"),
            coordinate_input_paths=("metrics/centroid_xy",),
            row_axis_name="observation",
            collection_axis=True,
            semantic_kind="validity",
        ),
        "metrics/bbox_valid": _scalar_spec(
            required["bbox_valid"],
            quantity="bounding_box_validity",
            units="1",
            operation="foreground_pixel_count_greater_than_zero_v1",
            axes=("observation", "subject_component"),
            coordinate_input_paths=("metrics/bbox_xyxy",),
            row_axis_name="observation",
            collection_axis=True,
            semantic_kind="validity",
        ),
        "available_channels": _scalar_spec(
            required["available_channels"],
            quantity="component_availability",
            units="1",
            operation="declared_refined_component_availability_v1",
            axes=("subject_component",),
            selected_collection_members=context.labels,
            semantic_kind="availability",
        ),
    }
    result.update({path: dict(spec) for path, spec in relation_specs.items()})

    for path, spec in sorted(coordinate_specs.items()):
        if path.endswith("/ellipse_params"):
            component = str(spec["component"])
            result[path.removesuffix("ellipse_params") + "ellipse_success"] = (
                _scalar_spec(
                    spec["validity_node"],
                    quantity="ellipse_fit_validity",
                    units="1",
                    operation="cv2_fit_ellipse_success_v1",
                    axes=("observation",),
                    coordinate_input_paths=(path,),
                    row_axis_name="observation",
                    selected_collection_members=(component,),
                    semantic_kind="validity",
                )
            )
        elif path.endswith("/sampled_contours/points_xy"):
            component = str(spec["component"])
            prefix = path.removesuffix("points_xy")
            result[prefix + "valid"] = _scalar_spec(
                spec["validity_node"],
                quantity="sampled_contour_validity",
                units="1",
                operation="sampled_contour_has_source_points_v1",
                axes=("observation",),
                coordinate_input_paths=(path,),
                row_axis_name="observation",
                selected_collection_members=(component,),
                semantic_kind="validity",
            )
            source_count = spec.get("companion_nodes", {}).get("source_point_count")
            if source_count is None:
                _fail(f"Sampled contour {path!r} lacks exact source_point_count.")
            result[prefix + "source_point_count"] = _scalar_spec(
                source_count,
                quantity="source_contour_point_count",
                units="1",
                operation="source_contour_point_count_before_arc_sampling_v1",
                axes=("observation",),
                coordinate_input_paths=(path,),
                row_axis_name="observation",
                selected_collection_members=(component,),
                semantic_kind="count",
            )

    for path, spec in sorted(ragged_specs.items()):
        component = str(spec["component"])
        prefix = path.removesuffix("points_xy")
        result[prefix + "ptr"] = _scalar_spec(
            spec["ptr_node"],
            quantity="packed_contour_start_index",
            units="1",
            operation="canonical_contiguous_packed_contour_offset_v1",
            axes=("observation",),
            row_axis_name="observation",
            selected_collection_members=(component,),
            semantic_kind="index",
        )
        result[prefix + "len"] = _scalar_spec(
            spec["len_node"],
            quantity="packed_contour_point_count",
            units="1",
            operation="canonical_packed_contour_length_v1",
            axes=("observation",),
            row_axis_name="observation",
            selected_collection_members=(component,),
            semantic_kind="count",
        )

    components = _child(context._run_group, "components", label="refined components")
    run_area = _array(required["area_px"], label="run-level area_px aliases")
    run_present = _array(
        required["mask_present"], label="run-level mask_present aliases"
    )
    for component_index, component in enumerate(context.labels):
        group = _child(components, component, label=f"refined component {component}")
        root_area = _child(group, "area_px", label=f"{component} area_px alias")
        root_present = _child(
            group, "mask_present", label=f"{component} mask_present alias"
        )
        _require_shape_dtype(
            root_area,
            shape=(rows,),
            dtype=np.dtype("float32"),
            label=f"{component} area_px alias",
        )
        _require_shape_dtype(
            root_present,
            shape=(rows,),
            dtype=np.dtype("bool"),
            label=f"{component} mask_present alias",
        )
        if not np.array_equal(
            _array(root_area, label=f"{component} area_px alias"),
            run_area[:, component_index],
        ) or not np.array_equal(
            _array(root_present, label=f"{component} mask_present alias"),
            run_present[:, component_index],
        ):
            _fail(
                f"Component {component!r} area/mask-present mirrors differ from exact "
                "run-level metric columns."
            )
        result[f"components/{component}/area_px"] = _scalar_spec(
            root_area,
            quantity="foreground_area",
            units="px^2",
            operation="exact_component_view_of_run_area_metric_v1",
            axes=("observation",),
            measurement_input_paths=("metrics/area_px",),
            row_axis_name="observation",
            selected_collection_members=(component,),
            semantic_kind="area",
            validity_node=root_present,
            validity_policy="zero_when_mask_absent_v1",
        )
        result[f"components/{component}/mask_present"] = _scalar_spec(
            root_present,
            quantity="mask_presence",
            units="1",
            operation="exact_component_view_of_run_presence_metric_v1",
            axes=("observation",),
            measurement_input_paths=("metrics/mask_present",),
            row_axis_name="observation",
            selected_collection_members=(component,),
            semantic_kind="validity",
        )

        for group_name, definitions in (
            ("metrics", _COMPONENT_METRIC_DEFINITIONS),
            ("finalization_metrics", _FINALIZATION_METRIC_DEFINITIONS),
        ):
            metric_group = _optional_child(group, group_name)
            if metric_group is None:
                continue
            unknown = set(_member_names(metric_group)) - set(definitions)
            if unknown:
                _fail(
                    f"Component {component!r} {group_name} contains unsupported scalar "
                    f"leaves {tuple(sorted(unknown))!r}."
                )
            for metric_name in _member_names(metric_group):
                node = _child(
                    metric_group,
                    metric_name,
                    label=f"{component} {group_name}/{metric_name}",
                )
                if not hasattr(node, "shape"):
                    _fail(
                        f"Component scalar leaf {component}/{group_name}/{metric_name} is not an array."
                    )
                shape = tuple(int(value) for value in node.shape)
                dtype = np.dtype(node.dtype)
                if shape != (rows,) or dtype.hasobject:
                    _fail(
                        f"Component scalar leaf {component}/{group_name}/{metric_name} "
                        "must be non-object and observation aligned."
                    )
                quantity, units, operation, semantic_kind = definitions[metric_name]
                result[f"components/{component}/{group_name}/{metric_name}"] = (
                    _scalar_spec(
                        node,
                        quantity=quantity,
                        units=units,
                        operation=operation,
                        axes=("observation",),
                        coordinate_input_paths=("masks_roi",),
                        row_axis_name="observation",
                        selected_collection_members=(component,),
                        semantic_kind=semantic_kind,
                        validity_node=root_present,
                        validity_policy="zero_or_nan_when_mask_absent_per_metric_v1",
                    )
                )
    return result


def _interpretation_record(
    context: BoundRefinedSubjectMaskCoordinateContext,
    logical_path: str,
    spec: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    validity_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    geometry = str(spec["geometry_type"])
    operation = {
        "masks_roi": "authoritative_refined_binary_roi_raster_v1",
        "metrics/centroid_xy": "mean_foreground_pixel_center_xy_v1",
        "metrics/bbox_xyxy": "foreground_half_open_pixel_edges_xyxy_v1",
    }.get(logical_path)
    if operation is None and logical_path.endswith("/ellipse_params"):
        operation = "cv2_fit_ellipse_from_authoritative_component_mask_v1"
    if operation is None and logical_path.endswith("/sampled_contours/points_xy"):
        operation = "closed_arc_length_sample_of_component_contour_v1"
    if logical_path == "relations/eye_pair/metrics/separation_px":
        operation = "euclidean_distance_between_refined_eye_centroids_v1"
    if operation is None:
        _fail(f"No exact refined geometry operation exists for {logical_path!r}.")
    record: dict[str, Any] = {
        "schema_id": REFINED_SUBJECT_MASK_ARRAY_INTERPRETATION_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "logical_path": logical_path,
        "array_path": canonical_node_path(spec["node"]),
        "array_payload": copy.deepcopy(dict(payload)),
        "surface_role": (
            "authoritative_pixels"
            if logical_path == "masks_roi"
            else "sealed_derived_geometry"
        ),
        "geometry_type": geometry,
        "coordinate_space": "roi_local_px",
        "pixel_convention": spec["pixel_convention"],
        "operation": operation,
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "component_labels": _record_pointer(context.component_labels),
        "source_authority": _record_pointer(context.source_authority),
        "refinement_authority": _record_pointer(context.refinement_authority),
        "component": spec.get("component"),
    }
    container = spec.get("container")
    if container is not None:
        record["container_evidence"] = {
            "path": canonical_node_path(container),
            "attrs": _exact_json_attrs(
                container,
                label=f"{logical_path} geometry container",
            ),
            "members": list(_member_names(container)),
        }
    if validity_payload is not None:
        record["validity"] = {
            "payload": copy.deepcopy(dict(validity_payload)),
            "false_value_policy": spec.get("invalid_value_policy")
            or (
                "zero_xy_is_invalid_sentinel_not_coordinate"
                if geometry == "point_xy"
                else "zero_geometry_is_invalid_sentinel_not_coordinate"
            ),
        }
    if "fit_result_diagnostics" in spec:
        record["fit_result_diagnostics"] = copy.deepcopy(
            dict(spec["fit_result_diagnostics"])
        )
    return record


def _ragged_record(
    context: BoundRefinedSubjectMaskCoordinateContext,
    logical_path: str,
    spec: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    node = spec["node"]
    ptr = spec["ptr_node"]
    length = spec["len_node"]
    return {
        "schema_id": REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "logical_path": logical_path,
        "array_path": canonical_node_path(node),
        "array_payload": copy.deepcopy(dict(payloads[logical_path])),
        "geometry_type": "points_xy",
        "coordinate_space": "roi_local_px",
        "components": ["x", "y"],
        "component_units": ["px", "px"],
        "origin": "top_left",
        "positive_directions": {"x": "right", "y": "down"},
        "pixel_convention": "pixel_center",
        "component": spec["component"],
        "container_evidence": {
            "path": canonical_node_path(spec["container"]),
            "attrs": _exact_json_attrs(
                spec["container"],
                label=f"{logical_path} ragged geometry container",
            ),
            "members": list(_member_names(spec["container"])),
        },
        "observation_row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "ragged_row_mapping": {
            "policy": "ptr_len_indexed_by_observation_row_v1",
            "ptr_payload": copy.deepcopy(dict(payloads[f"{logical_path}@ptr"])),
            "len_payload": copy.deepcopy(dict(payloads[f"{logical_path}@len"])),
            "point_count": int(spec["point_count"]),
            "ptr_path": canonical_node_path(ptr),
            "len_path": canonical_node_path(length),
        },
        "source_camera_overlay": {
            "status": "requires_transform",
            "direction": "roi_local_px_to_source_camera_image_px",
            "transform_refs": [
                {
                    "record_ref": item.record_ref,
                    "record_sha256": item.record_sha256,
                }
                for item in context.pixel_center_chain.transform_records
            ],
        },
        "reference_frame": {
            "record_ref": context.pixel_center_frame.record_ref,
            "record_sha256": context.pixel_center_frame.record_sha256,
            "width": int(context.pixel_center_frame.endpoint.width),
            "height": int(context.pixel_center_frame.endpoint.height),
            "units": "px",
        },
        "source_authority": _record_pointer(context.source_authority),
        "refinement_authority": _record_pointer(context.refinement_authority),
    }


def _optional_payloads(
    canonical: Mapping[str, Mapping[str, Any]],
    ragged: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path, spec in canonical.items():
        result[path] = _payload(spec["node"])
        validity = spec.get("validity_node")
        if validity is not None:
            result[f"{path}@validity"] = _payload(validity)
        for name, node in dict(spec.get("companion_nodes") or {}).items():
            result[f"{path}@companion:{name}"] = _payload(node)
    for path, spec in ragged.items():
        result[path] = _payload(spec["node"])
        result[f"{path}@ptr"] = _payload(spec["ptr_node"])
        result[f"{path}@len"] = _payload(spec["len_node"])
    return result


def _bind_interpretations(
    context: BoundRefinedSubjectMaskCoordinateContext,
    specs: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    stamp: bool,
) -> dict[str, BoundCoordinateRecord]:
    result: dict[str, BoundCoordinateRecord] = {}
    for path, spec in sorted(specs.items()):
        validity_payload = payloads.get(f"{path}@validity")
        expected = _interpretation_record(
            context,
            path,
            spec,
            payload=payloads[path],
            validity_payload=validity_payload,
        )
        node = spec["node"]
        value = (
            stamp_and_bind_persisted_coordinate_record(
                node,
                expected,
                attr_name=REFINED_SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR,
            )
            if stamp
            else bind_persisted_coordinate_record(
                node,
                attr_name=REFINED_SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR,
            )
        )
        if value.record != expected:
            _fail(f"Refined interpretation for {path!r} differs from live evidence.")
        result[path] = value
    return result


def _bind_ragged_records(
    context: BoundRefinedSubjectMaskCoordinateContext,
    specs: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    stamp: bool,
) -> dict[str, BoundCoordinateRecord]:
    result: dict[str, BoundCoordinateRecord] = {}
    for path, spec in sorted(specs.items()):
        expected = _ragged_record(context, path, spec, payloads)
        node = spec["node"]
        value = (
            stamp_and_bind_persisted_coordinate_record(
                node,
                expected,
                attr_name=REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_ATTR,
            )
            if stamp
            else bind_persisted_coordinate_record(
                node,
                attr_name=REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_ATTR,
            )
        )
        if value.record != expected:
            _fail(f"Refined ragged geometry for {path!r} differs from live evidence.")
        result[path] = value
    return result


def _cache_inventory(run: Any, masks_payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("mask_bitpacked", "mask_rle"):
        node = _optional_child(run, name)
        if node is None:
            continue
        if hasattr(node, "shape"):
            _fail(f"{name} must be a derived cache group, not an authoritative array.")
        attrs = getattr(node, "attrs", {})
        if (
            attrs.get("surface_role")
            not in {
                "derived_display_cache",
                "derived_archive_cache",
                "derived_display_archive_cache",
            }
            or attrs.get("authoritative_pixels") is not False
        ):
            _fail(
                f"{name} must explicitly declare derived/non-authoritative cache semantics."
            )
        result[name] = {
            "path": canonical_node_path(node),
            "surface_role": attrs.get("surface_role"),
            "authoritative_pixels": False,
            "source_masks_roi_payload": copy.deepcopy(dict(masks_payload)),
            "exact_contents": _recursive_namespace_inventory(
                node,
                label=f"derived cache {name}",
            ),
        }
    return result


_KNOWN_COMPONENT_ROOT_ARRAYS = frozenset(
    {
        "area_px",
        "edit_applied",
        "manual_override",
        "mask_present",
        "reason_bytes",
        "source_row_fingerprint",
        "source_row_stale",
        "row_revision",
        "row_updated_at_utc_bytes",
        "row_update_reason_bytes",
        "source_seed_masks_roi",
    }
)
_KNOWN_COMPONENT_ROOT_GROUPS = frozenset(
    {
        "contours",
        "finalization_metrics",
        "geometry",
        "metrics",
        "provenance",
        "qc",
        "sampled_contours",
    }
)
_KNOWN_RUN_ROOT_ARRAY_ROLES: Mapping[str, str] = {
    "masks_roi": "authoritative_refined_binary_roi_raster",
    "available_channels": "subject_component_availability",
    "edit_applied": "draft_edit_state_not_scientific_measurement",
    "source_crop_row_ids": "source_crop_row_identity",
    "instance_key": "observation_instance_identity",
    "source_acquisition_frame_index": "source_temporal_identity",
    "frame_row_offsets": "source_frame_to_observation_row_csr_index",
    "source_crop_xywh": "roi_to_source_camera_placement",
}
_KNOWN_RUN_ROOT_GROUPS = frozenset(
    {
        "components",
        "coordinate_frames",
        "mask_bitpacked",
        "mask_rle",
        "metrics",
        "relations",
    }
)


def _is_explicit_non_geometry(node: Any) -> bool:
    attrs = getattr(node, "attrs", {})
    return (
        attrs.get("surface_role") == "explicit_non_geometry"
        and attrs.get("geometry_semantics") == "none"
    )


def _closed_world_structure_inventory(
    context: BoundRefinedSubjectMaskCoordinateContext,
) -> dict[str, Any]:
    """Reject undeclared component/relation namespaces and bind root surfaces."""

    run = context._run_group
    root_arrays: dict[str, Any] = {}
    root_groups: dict[str, Any] = {}
    for name in _member_names(run):
        node = _child(run, name, label=f"refined run root member {name}")
        if hasattr(node, "shape"):
            role = _KNOWN_RUN_ROOT_ARRAY_ROLES.get(name)
            if role is None:
                _fail(
                    f"Refined run contains unsupported root array {name!r}; its "
                    "scientific semantics are not declared."
                )
            root_arrays[name] = {
                "semantic_role": role,
                "payload": _payload(node),
                "producer_attrs": _scientific_attrs(
                    node,
                    label=f"refined run root array {name}",
                ),
            }
            continue
        if name not in _KNOWN_RUN_ROOT_GROUPS:
            _fail(f"Refined run contains unsupported root namespace {name!r}.")
        root_groups[name] = {
            "semantic_role": {
                "components": "typed_subject_component_namespaces",
                "coordinate_frames": "roi_coordinate_and_transform_authorities",
                "mask_bitpacked": "derived_non_authoritative_cache",
                "mask_rle": "derived_non_authoritative_cache",
                "metrics": "typed_run_scientific_metrics",
                "relations": "typed_cross_component_relations",
            }[name],
            "producer_attrs": _scientific_attrs(
                node,
                label=f"refined run root namespace {name}",
            ),
            "members": list(_member_names(node)),
        }
    components_parent = _optional_child(run, "components")
    result_components: dict[str, Any] = {}
    if components_parent is not None:
        component_names = set(_member_names(components_parent))
        undeclared = sorted(component_names - set(context.labels))
        if undeclared:
            _fail(
                "Refined components contains undeclared component namespaces: "
                f"{tuple(undeclared)!r}."
            )
    for component in context.labels:
        group = _component_group(run, component)
        if group is None:
            result_components[component] = {"status": "absent"}
            continue
        arrays: dict[str, Any] = {}
        groups: dict[str, Any] = {}
        for name in _member_names(group):
            node = _child(group, name, label=f"{component} root member {name}")
            if hasattr(node, "shape"):
                if (
                    name not in _KNOWN_COMPONENT_ROOT_ARRAYS
                    and not _is_explicit_non_geometry(node)
                ):
                    _fail(
                        f"Refined component {component!r} contains undocumented root "
                        f"array {name!r}; coordinate/geometry semantics cannot be inferred."
                    )
                shape = tuple(int(value) for value in node.shape)
                if not shape or shape[0] != context.row_identity.leading_dimension:
                    _fail(
                        f"Refined component root array {component}/{name} is not aligned "
                        "to the observation row identity."
                    )
                arrays[name] = {
                    "classification": (
                        "retained_source_seed_diagnostic_raster"
                        if name == "source_seed_masks_roi"
                        else (
                            "known_non_coordinate_component_state"
                            if name in _KNOWN_COMPONENT_ROOT_ARRAYS
                            else "explicit_non_geometry"
                        )
                    ),
                    "payload": _payload(node),
                    "attrs": _scientific_attrs(
                        node,
                        label=f"{component}/{name} root array",
                    ),
                }
                continue
            if name in _KNOWN_COMPONENT_ROOT_GROUPS:
                groups[name] = {
                    "classification": "known_typed_namespace",
                    "exact_contents": _recursive_namespace_inventory(
                        node,
                        label=f"{component}/{name}",
                    ),
                }
                continue
            if not _is_explicit_non_geometry(node):
                _fail(
                    f"Refined component {component!r} contains undocumented root "
                    f"namespace {name!r}."
                )
            classified_arrays: dict[str, Any] = {}
            for child_name in _member_names(node):
                child = _child(
                    node,
                    child_name,
                    label=f"{component}/{name}/{child_name}",
                )
                if not hasattr(child, "shape") or not _is_explicit_non_geometry(child):
                    _fail(
                        f"Explicit non-geometry namespace {component}/{name} may contain "
                        "only explicitly classified arrays."
                    )
                classified_arrays[child_name] = _payload(child)
            groups[name] = {
                "classification": "explicit_non_geometry",
                "attrs": _exact_json_attrs(node, label=f"{component}/{name}"),
                "arrays": classified_arrays,
            }
        result_components[component] = {
            "status": "present",
            "producer_attrs": _scientific_attrs(
                group,
                label=f"refined component {component}",
            ),
            "arrays": arrays,
            "groups": groups,
        }

    relations = _optional_child(run, "relations")
    relation_inventory: dict[str, Any] = {}
    if relations is not None:
        relation_names = set(_member_names(relations))
        unknown = sorted(relation_names - {"eye_pair"})
        if unknown:
            _fail(
                "Refined run contains unknown relation namespaces: "
                f"{tuple(unknown)!r}."
            )
        if "eye_pair" in relation_names:
            eye_pair = _child(relations, "eye_pair", label="eye_pair relation")
            members = set(_member_names(eye_pair))
            if members - {"metrics"}:
                _fail(
                    "Refined eye_pair relation contains unsupported root members: "
                    f"{tuple(sorted(members - {'metrics'}))!r}."
                )
            relation_inventory["eye_pair"] = {
                "exact_contents": _recursive_namespace_inventory(
                    eye_pair,
                    label="eye_pair relation",
                ),
            }
    return {
        "policy": "closed_world_exact_component_and_relation_namespaces_v2",
        "run_root": {
            "arrays": root_arrays,
            "groups": root_groups,
        },
        "components_parent_attrs": (
            _scientific_attrs(
                components_parent,
                label="refined components parent",
            )
            if components_parent is not None
            else {}
        ),
        "relations_parent_attrs": (
            _scientific_attrs(relations, label="refined relations parent")
            if relations is not None
            else {}
        ),
        "components": result_components,
        "relations": relation_inventory,
    }


def _component_qc_inventory_record(
    context: BoundRefinedSubjectMaskCoordinateContext,
) -> dict[str, Any]:
    """Bind every component QC child exactly, including explicit absence."""

    components_parent = _optional_child(context._run_group, "components")
    rows = context.row_identity.leading_dimension
    components: dict[str, Any] = {}
    for component in context.labels:
        component_group = (
            _optional_child(components_parent, component)
            if components_parent is not None
            else None
        )
        qc = (
            _optional_child(component_group, "qc")
            if component_group is not None
            else None
        )
        if qc is None:
            components[component] = {
                "status": "absent",
                "path": f"{context.run_path}/components/{component}/qc",
            }
            continue
        if hasattr(qc, "shape"):
            _fail(f"{component} QC authority must be a group, not an array.")
        names = _member_names(qc)
        if not names:
            _fail(f"{component} QC group is present but contains no typed arrays.")
        arrays: dict[str, Any] = {}
        for name in names:
            node = _child(qc, name, label=f"{component} QC {name}")
            if not hasattr(node, "shape"):
                _fail(f"{component} QC contains unsupported nested group {name!r}.")
            shape = tuple(int(value) for value in node.shape)
            if not shape or shape[0] != rows:
                _fail(
                    f"{component} QC array {name!r} is not aligned to the refined row identity."
                )
            try:
                dtype = np.dtype(node.dtype)
            except (AttributeError, TypeError) as exc:
                _fail(f"{component} QC array {name!r} lacks exact dtype: {exc}.")
            if dtype.hasobject:
                _fail(f"{component} QC array {name!r} cannot use object dtype.")
            arrays[name] = _payload(node)
        components[component] = {
            "status": "present",
            "path": canonical_node_path(qc),
            "attrs": _exact_json_attrs(qc, label=f"{component} QC"),
            "arrays": arrays,
        }
    return {
        "schema_id": REFINED_SUBJECT_MASK_COMPONENT_QC_INVENTORY_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "component_labels": _record_pointer(context.component_labels),
        "policy": "closed_world_exact_component_qc_arrays_and_attrs_v1",
        "components": components,
    }


def _inventory_record(
    context: BoundRefinedSubjectMaskCoordinateContext,
    specs: Mapping[str, Mapping[str, Any]],
    ragged_specs: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    interpretations: Mapping[str, BoundCoordinateRecord],
    ragged_records: Mapping[str, BoundCoordinateRecord],
    component_qc_inventory: BoundCoordinateRecord,
    structure_inventory: Mapping[str, Any],
    measurement_specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_id": REFINED_SUBJECT_MASK_SURFACE_INVENTORY_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "component_labels": _record_pointer(context.component_labels),
        "source_authority": _record_pointer(context.source_authority),
        "refinement_authority": _record_pointer(context.refinement_authority),
        "assignment_keypoint_dependency": _assignment_dependency_summary(
            context.assignment_keypoint_authority
        ),
        "component_qc_inventory": _record_pointer(component_qc_inventory),
        "closed_world_structure": copy.deepcopy(dict(structure_inventory)),
        "authoritative_raster": "masks_roi",
        "required_geometry": [
            "masks_roi",
            "metrics/centroid_xy",
            "metrics/bbox_xyxy",
        ],
        "canonical_geometry": {
            path: {
                "payload": copy.deepcopy(dict(payloads[path])),
                "interpretation": _record_pointer(interpretations[path]),
            }
            for path in sorted(specs)
        },
        "ragged_geometry": {
            path: {
                "payload": copy.deepcopy(dict(payloads[path])),
                "row_mapping": {
                    "ptr": copy.deepcopy(dict(payloads[f"{path}@ptr"])),
                    "len": copy.deepcopy(dict(payloads[f"{path}@len"])),
                },
                "interpretation": _record_pointer(ragged_records[path]),
            }
            for path in sorted(ragged_specs)
        },
        "measurement_surfaces": {
            path: {
                "payload": _payload(spec["node"]),
                "semantic_kind": str(spec["semantic_kind"]),
                "quantity": str(spec["quantity"]),
                "units": str(spec["units"]),
                "validity_payload": (
                    _payload(spec["validity_node"])
                    if spec.get("validity_node") is not None
                    else None
                ),
            }
            for path, spec in sorted(measurement_specs.items())
        },
        "derived_companions": {
            path: copy.deepcopy(dict(payload))
            for path, payload in sorted(payloads.items())
            if path not in specs
            and path not in ragged_specs
            and not path.endswith("@ptr")
            and not path.endswith("@len")
        },
        "compact_caches": _cache_inventory(
            context._run_group,
            payloads["masks_roi"],
        ),
    }


def _measurement_authority_record(
    context: BoundRefinedSubjectMaskCoordinateContext,
    *,
    inventory: BoundCoordinateRecord,
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    surfaces: dict[str, Any] = {}
    for path, spec in sorted(specs.items()):
        entry: dict[str, Any] = {
            "payload": _payload(spec["node"]),
            "quantity": str(spec["quantity"]),
            "units": str(spec["units"]),
            "operation": str(spec["operation"]),
            "axis_order": list(spec["axes"]),
            "semantic_kind": str(spec["semantic_kind"]),
            "coordinate_input_paths": list(spec.get("coordinate_input_paths", ())),
            "measurement_input_paths": list(spec.get("measurement_input_paths", ())),
            "selected_collection_members": list(
                spec.get("selected_collection_members", ())
            ),
        }
        validity_node = spec.get("validity_node")
        if validity_node is not None:
            entry["validity"] = {
                "payload": _payload(validity_node),
                "policy": str(spec["validity_policy"]),
            }
        surfaces[path] = entry
    return {
        "schema_id": REFINED_SUBJECT_MASK_MEASUREMENT_AUTHORITY_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "policy": "closed_world_array_specific_measurement_semantics_v1",
        "run_path": context.run_path,
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "component_labels": _record_pointer(context.component_labels),
        "source_authority": _record_pointer(context.source_authority),
        "refinement_authority": _record_pointer(context.refinement_authority),
        "surface_inventory": _record_pointer(inventory),
        "authoritative_mask_payload": _payload(
            _child(
                context._run_group,
                "masks_roi",
                label="measurement authoritative masks_roi",
            )
        ),
        "surfaces": surfaces,
    }


def _measurement_bindings(
    context: BoundRefinedSubjectMaskCoordinateContext,
    specs: Mapping[str, Mapping[str, Any]],
    coordinate_bindings: Mapping[str, BoundCanonicalCoordinateDescriptor],
    measurement_authority: BoundCoordinateRecord,
    *,
    load: bool,
) -> dict[str, BoundCoordinateRecord]:
    result: dict[str, BoundCoordinateRecord] = {}
    remaining = dict(specs)
    while remaining:
        progressed = False
        for path in sorted(tuple(remaining)):
            spec = remaining[path]
            input_paths = tuple(spec.get("measurement_input_paths", ()))
            if any(input_path not in result for input_path in input_paths):
                continue
            coordinate_paths = tuple(spec.get("coordinate_input_paths", ()))
            missing_coordinates = tuple(
                input_path
                for input_path in coordinate_paths
                if input_path not in coordinate_bindings
            )
            if missing_coordinates:
                _fail(
                    f"Measurement {path!r} references missing coordinate inputs "
                    f"{missing_coordinates!r}."
                )
            try:
                expected = build_array_measurement_descriptor(
                    spec["node"],
                    quantity=spec["quantity"],
                    units=spec["units"],
                    operation=spec["operation"],
                    axes=spec["axes"],
                    coordinate_inputs=tuple(
                        coordinate_bindings[input_path]
                        for input_path in coordinate_paths
                    ),
                    measurement_inputs=(
                        *(result[input_path] for input_path in input_paths),
                        *tuple(spec.get("record_inputs", ())),
                    ),
                    row_identity=context.row_identity,
                    collection=context.component_labels,
                    measurement_authority=measurement_authority,
                    derivation=context.refinement_authority,
                    row_axis_name=spec.get("row_axis_name"),
                    collection_axis_name=(
                        "subject_component" if spec.get("collection_axis") else None
                    ),
                    collection_axis_role=(
                        "subject_component" if spec.get("collection_axis") else None
                    ),
                    validity_node=spec.get("validity_node"),
                    validity_policy=spec.get("validity_policy"),
                    selected_collection_members=spec.get(
                        "selected_collection_members",
                        (),
                    ),
                    semantic_kind=spec.get("semantic_kind"),
                )
                result[path] = (
                    load_bound_array_measurement_descriptor(
                        spec["node"],
                        expected_record=expected,
                    )
                    if load
                    else stamp_and_bind_array_measurement_descriptor(
                        spec["node"],
                        expected,
                    )
                )
            except Exception as exc:
                _fail(f"Measurement descriptor for {path!r} is invalid: {exc}.")
            del remaining[path]
            progressed = True
        if not progressed:
            _fail(
                "Measurement descriptor dependencies contain a cycle or missing input: "
                f"{tuple(sorted(remaining))!r}."
            )
    return result


def _scientific_manifest_record(
    context: BoundRefinedSubjectMaskCoordinateContext,
    *,
    inventory: BoundCoordinateRecord,
    specs: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    measurement_authority: BoundCoordinateRecord,
    measurements: Mapping[str, BoundCoordinateRecord],
) -> dict[str, Any]:
    run = context._run_group
    table_nodes = {
        "available_channels": _child(
            run, "available_channels", label="available channels"
        ),
        "source_crop_row_ids": _child(
            run, "source_crop_row_ids", label="source crop row ids"
        ),
        "instance_key": _child(run, "instance_key", label="instance key"),
        "source_acquisition_frame_index": _child(
            run,
            "source_acquisition_frame_index",
            label="source acquisition frame index",
        ),
        "frame_row_offsets": _child(
            run,
            "frame_row_offsets",
            label="source frame to observation row offsets",
        ),
        "source_crop_xywh": _child(
            run, "source_crop_xywh", label="source crop placement"
        ),
    }
    table_payloads = {name: _payload(node) for name, node in table_nodes.items()}
    for logical_path in (
        "metrics/mask_present",
        "metrics/area_px",
        "metrics/centroid_xy",
        "metrics/centroid_xy@validity",
        "metrics/bbox_xyxy",
        "metrics/bbox_xyxy@validity",
    ):
        table_payloads[logical_path] = copy.deepcopy(dict(payloads[logical_path]))
    table_payloads["masks_roi"] = copy.deepcopy(dict(payloads["masks_roi"]))
    return {
        "schema_id": REFINED_SUBJECT_MASK_SCIENTIFIC_MANIFEST_SCHEMA_ID,
        "schema_version": REFINED_SUBJECT_MASK_SCHEMA_VERSION,
        "policy": "typed_sealed_scientific_subset_only_v1",
        "surface_inventory": _record_pointer(inventory),
        "row_identity": {
            "record_ref": context.row_identity.record_ref,
            "record_sha256": context.row_identity.record_sha256,
        },
        "table_arrays": table_payloads,
        "coordinate_descriptor_paths": sorted(specs),
        "measurement_authority": _record_pointer(measurement_authority),
        "measurement_descriptor_paths": {
            path: _record_pointer(value) for path, value in sorted(measurements.items())
        },
        "canonical_table_exclusions": {
            "run_arrays": [
                "detection_source",
                "edit_applied",
                "frame_counts",
                "frame_indices",
                "reason",
                "reason_bytes",
                "source_detect_row_index",
                "source_refined_row_ids",
            ],
            "component_namespaces": "not_exposed_by_canonical_table_reader",
            "relation_namespaces": "not_exposed_by_canonical_table_reader",
            "rationale": (
                "mutable review/revision/reason/staleness state is historical inspection "
                "data, not part of the canonical scientific table claim"
            ),
        },
    }


def _collection_axis(
    context: BoundRefinedSubjectMaskCoordinateContext,
) -> CanonicalCollectionAxis:
    return CanonicalCollectionAxis(
        axis=1,
        role="subject_component",
        cardinality=len(context.labels),
        label_authority=DigestBoundCoordinateRecordRef(
            record_ref=context.component_labels.record_ref,
            record_sha256=context.component_labels.record_sha256,
        ),
    )


def _bindings(
    context: BoundRefinedSubjectMaskCoordinateContext,
    specs: Mapping[str, Mapping[str, Any]],
    inventory: BoundCoordinateRecord,
    interpretations: Mapping[str, BoundCoordinateRecord],
    ragged_records: Mapping[str, BoundCoordinateRecord],
    *,
    load: bool,
) -> dict[str, BoundCanonicalCoordinateDescriptor]:
    result: dict[str, BoundCanonicalCoordinateDescriptor] = {}
    collection = _collection_axis(context)
    ragged_lineage = tuple(ragged_records[path] for path in sorted(ragged_records))
    for path, spec in sorted(specs.items()):
        lineage = (
            context.component_labels,
            context.source_authority,
            context.refinement_authority,
            context.context_record,
            interpretations[path],
            inventory,
            *ragged_lineage,
        )
        kwargs = {
            "row_identity": context.row_identity,
            "reference_frame_authority": spec["frame"],
            "transform_chain": spec["chain"],
            "lineage_records": lineage,
        }
        result[path] = (
            load_bound_canonical_coordinate_descriptor(spec["node"], **kwargs)
            if load
            else build_bound_canonical_coordinate_descriptor(
                spec["node"],
                profile_id="roi_local_px.top_left_y_down.v1",
                geometry_type=spec["geometry_type"],
                components=spec["components"],
                component_units=spec["component_units"],
                pixel_convention=spec["pixel_convention"],
                row_identity=context.row_identity,
                reference_frame_authority=spec["frame"],
                source_camera_overlay_status=spec["overlay"],
                transform_chain=spec["chain"],
                lineage_records=lineage,
                collection_axis=(
                    collection if bool(spec.get("collection_axis")) else None
                ),
            )
        )
    return result


@dataclass(frozen=True, init=False)
class BoundRefinedSubjectMaskCoordinateSurfaces:
    descriptors: Mapping[str, BoundCanonicalCoordinateDescriptor]
    context: BoundRefinedSubjectMaskCoordinateContext = field(repr=False)
    inventory: BoundCoordinateRecord = field(repr=False)
    component_qc_inventory: BoundCoordinateRecord = field(repr=False)
    measurement_authority: BoundCoordinateRecord = field(repr=False)
    measurements: Mapping[str, BoundCoordinateRecord] = field(repr=False)
    scientific_manifest: BoundCoordinateRecord = field(repr=False)
    interpretations: Mapping[str, BoundCoordinateRecord] = field(repr=False)
    ragged_geometry: Mapping[str, BoundCoordinateRecord] = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self, *, _verification_seal: object | None = None, **values: Any
    ) -> None:
        if _verification_seal is not _BOUND_SURFACES_SEAL:
            _fail("Refined coordinate surfaces cannot be constructed directly.")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)

    @property
    def masks_roi(self) -> BoundCanonicalCoordinateDescriptor:
        return self.descriptors["masks_roi"]

    @property
    def centroid_xy(self) -> BoundCanonicalCoordinateDescriptor:
        return self.descriptors["metrics/centroid_xy"]

    @property
    def bbox_xyxy(self) -> BoundCanonicalCoordinateDescriptor:
        return self.descriptors["metrics/bbox_xyxy"]


@dataclass(frozen=True, init=False)
class RefinedSubjectMaskCoordinatePublicationCheckpoint:
    run_path: str
    publication_owner: str
    _root: Any = field(repr=False, compare=False)
    _paths: tuple[str, ...] = field(repr=False, compare=False)
    _attrs: tuple[dict[str, Any], ...] = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(
        self,
        *,
        run_path: str,
        publication_owner: str,
        root: Any,
        paths: tuple[str, ...],
        attrs: tuple[dict[str, Any], ...],
        _verification_seal: object | None = None,
    ) -> None:
        if _verification_seal is not _CHECKPOINT_SEAL:
            _fail("Refined publication checkpoints cannot be constructed directly.")
        object.__setattr__(self, "run_path", run_path)
        object.__setattr__(self, "publication_owner", publication_owner)
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_paths", paths)
        object.__setattr__(self, "_attrs", attrs)
        object.__setattr__(self, "_seal", _verification_seal)


def _publication_targets(
    context: BoundRefinedSubjectMaskCoordinateContext,
    required: Mapping[str, Any],
    optional: Mapping[str, Mapping[str, Any]],
    ragged: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, ...]:
    targets: list[Any] = [context._run_group, *required.values()]
    for spec in optional.values():
        targets.append(spec["node"])
        if spec.get("validity_node") is not None:
            targets.append(spec["validity_node"])
    for spec in ragged.values():
        targets.extend((spec["node"], spec["ptr_node"], spec["len_node"]))
    for spec in measurements.values():
        targets.append(spec["node"])
        if spec.get("validity_node") is not None:
            targets.append(spec["validity_node"])
    for name in ("mask_bitpacked", "mask_rle"):
        cache = _optional_child(context._run_group, name)
        if cache is not None:
            targets.append(cache)
    return tuple(targets)


def capture_refined_subject_mask_coordinate_publication_checkpoint(
    root: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
) -> RefinedSubjectMaskCoordinatePublicationCheckpoint:
    path = _canonical_path(
        run_path,
        prefix="refined_subject_masks_runs/",
        label="refined subject-mask rowset",
    )
    context = _load_refined_subject_mask_coordinate_context(
        root,
        path,
        require_complete=False,
        expected_selector_eligible=False,
        expected_publication_owner=expected_publication_owner,
    )
    required, _payloads = _scan_required_surfaces(context)
    _closed_world_structure_inventory(context)
    optional, ragged, relation_measurements = _validate_optional_geometry(context)
    measurements = _measurement_specs(
        context,
        required,
        {**_required_geometry_specs(context, required), **optional},
        ragged,
        relation_measurements,
    )
    targets, attrs = _attrs_snapshot(
        *_publication_targets(context, required, optional, ragged, measurements)
    )
    return RefinedSubjectMaskCoordinatePublicationCheckpoint(
        run_path=path,
        publication_owner=expected_publication_owner,
        root=root,
        paths=targets,
        attrs=attrs,
        _verification_seal=_CHECKPOINT_SEAL,
    )


def rollback_refined_subject_mask_coordinate_publication(
    checkpoint: RefinedSubjectMaskCoordinatePublicationCheckpoint,
) -> None:
    if (
        type(checkpoint) is not RefinedSubjectMaskCoordinatePublicationCheckpoint
        or checkpoint._seal is not _CHECKPOINT_SEAL
    ):
        _fail("A sealed refined publication checkpoint is required.")
    _restore_attrs(
        checkpoint._root,
        checkpoint._paths,
        checkpoint._attrs,
        run_path=checkpoint.run_path,
        owner=checkpoint.publication_owner,
    )


def _surface_evidence(
    context: BoundRefinedSubjectMaskCoordinateContext,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    structure_inventory = _closed_world_structure_inventory(context)
    optional, ragged, relation_measurements = _validate_optional_geometry(context)
    # Fail on undeclared namespaces and malformed optional geometry before the
    # expensive dense-mask/metric equivalence scan. Structural incompatibility
    # must not consume minutes of payload I/O before it is reported.
    required_nodes, required_payloads = _scan_required_surfaces(context)
    specs = _required_geometry_specs(context, required_nodes)
    specs.update(optional)
    measurements = _measurement_specs(
        context,
        required_nodes,
        specs,
        ragged,
        relation_measurements,
    )
    payloads: dict[str, dict[str, Any]] = {
        "masks_roi": required_payloads["masks_roi"],
        "metrics/centroid_xy": required_payloads["centroid_xy"],
        "metrics/centroid_xy@validity": required_payloads["centroid_valid"],
        "metrics/bbox_xyxy": required_payloads["bbox_xyxy"],
        "metrics/bbox_xyxy@validity": required_payloads["bbox_valid"],
        "metrics/mask_present": required_payloads["mask_present"],
        "metrics/area_px": required_payloads["area_px"],
        "available_channels": required_payloads["available_channels"],
    }
    payloads.update(_optional_payloads(optional, ragged))
    return required_nodes, specs, ragged, measurements, payloads, structure_inventory


def _publish_refined_subject_mask_coordinate_surfaces_impl(
    root: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    """Seal all refined geometry while the child remains running/ineligible."""

    path = _canonical_path(
        run_path,
        prefix="refined_subject_masks_runs/",
        label="refined subject-mask rowset",
    )
    context = _load_refined_subject_mask_coordinate_context(
        root,
        path,
        require_complete=False,
        expected_selector_eligible=False,
        expected_publication_owner=expected_publication_owner,
    )
    (
        required,
        specs,
        ragged_specs,
        measurement_specs,
        payloads,
        structure_inventory,
    ) = _surface_evidence(context)
    targets, attrs = _attrs_snapshot(
        *_publication_targets(
            context,
            required,
            specs,
            ragged_specs,
            measurement_specs,
        )
    )
    checkpoint = RefinedSubjectMaskCoordinatePublicationCheckpoint(
        run_path=path,
        publication_owner=expected_publication_owner,
        root=root,
        paths=targets,
        attrs=attrs,
        _verification_seal=_CHECKPOINT_SEAL,
    )

    def authorize() -> Any:
        return _fresh_owned_ineligible_run(
            root,
            path,
            owner=expected_publication_owner,
            statuses=(RUN_STATUS_RUNNING,),
            label="Refined coordinate publication target",
        )

    try:
        authorize()
        interpretations = _bind_interpretations(
            context,
            specs,
            payloads,
            stamp=True,
        )
        authorize()
        ragged_records = _bind_ragged_records(
            context,
            ragged_specs,
            payloads,
            stamp=True,
        )
        active = authorize()
        component_qc_inventory = stamp_and_bind_persisted_coordinate_record(
            active,
            _component_qc_inventory_record(context),
            attr_name=REFINED_SUBJECT_MASK_COMPONENT_QC_INVENTORY_ATTR,
        )
        active = authorize()
        inventory = stamp_and_bind_persisted_coordinate_record(
            active,
            _inventory_record(
                context,
                specs,
                ragged_specs,
                payloads,
                interpretations,
                ragged_records,
                component_qc_inventory,
                structure_inventory,
                measurement_specs,
            ),
            attr_name=REFINED_SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
        )
        authorize()
        descriptors = _bindings(
            context,
            specs,
            inventory,
            interpretations,
            ragged_records,
            load=False,
        )
        stamp_bound_canonical_coordinate_descriptors(descriptors.values())
        active = authorize()
        measurement_authority = stamp_and_bind_persisted_coordinate_record(
            active,
            _measurement_authority_record(
                context,
                inventory=inventory,
                specs=measurement_specs,
            ),
            attr_name=REFINED_SUBJECT_MASK_MEASUREMENT_AUTHORITY_ATTR,
        )
        authorize()
        measurements = _measurement_bindings(
            context,
            measurement_specs,
            descriptors,
            measurement_authority,
            load=False,
        )
        active = authorize()
        scientific_manifest = stamp_and_bind_persisted_coordinate_record(
            active,
            _scientific_manifest_record(
                context,
                inventory=inventory,
                specs=specs,
                payloads=payloads,
                measurement_authority=measurement_authority,
                measurements=measurements,
            ),
            attr_name=REFINED_SUBJECT_MASK_SCIENTIFIC_MANIFEST_ATTR,
        )
        active = authorize()
        active.attrs["coordinate_contract"] = "canonical_v2"
        if active.attrs.get("coordinate_contract") != "canonical_v2":
            _fail("Refined coordinate contract did not persist exactly.")
        return BoundRefinedSubjectMaskCoordinateSurfaces(
            descriptors=descriptors,
            context=context,
            inventory=inventory,
            component_qc_inventory=component_qc_inventory,
            measurement_authority=measurement_authority,
            measurements=measurements,
            scientific_manifest=scientific_manifest,
            interpretations=interpretations,
            ragged_geometry=ragged_records,
            _verification_seal=_BOUND_SURFACES_SEAL,
        )
    except BaseException as exc:
        try:
            rollback_refined_subject_mask_coordinate_publication(checkpoint)
        except BaseException as rollback_exc:  # pragma: no cover - hostile store
            raise RefinedSubjectMaskCoordinatePublicationError(
                "Refined coordinate publication failed and rollback was incomplete: "
                f"{rollback_exc}."
            ) from exc
        raise


@proof_verification_operation
def publish_refined_subject_mask_coordinate_surfaces(
    root: Any,
    run_path: str,
    *,
    expected_publication_owner: str,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    with _payload_cache_scope():
        return _publish_refined_subject_mask_coordinate_surfaces_impl(
            root,
            run_path,
            expected_publication_owner=expected_publication_owner,
        )


def _receipt_path_under_run(
    value: Any,
    *,
    run_path: str,
    label: str,
) -> str:
    """Return a receipt path relative to its exact successor run."""

    if type(value) is not str or not value:
        _fail(f"{label} must be one nonempty path.")
    normalized = value.strip("/")
    marker = run_path.strip("/")
    if normalized == marker:
        return ""
    suffix = f"{marker}/"
    if normalized.startswith(suffix):
        return normalized[len(suffix) :]
    qualified = f"/{marker}/"
    index = normalized.find(qualified)
    if index >= 0:
        return normalized[index + len(qualified) :]
    if normalized.endswith(f"/{marker}"):
        return ""
    _fail(f"{label} leaves the exact successor run {run_path!r}.")


def _receipt_node(
    root: Any,
    *,
    run_path: str,
    path: Any,
    label: str,
) -> Any:
    relative = _receipt_path_under_run(path, run_path=run_path, label=label)
    return _node(
        root,
        run_path if not relative else f"{run_path}/{relative}",
        label=label,
    )


def _receipt_array_node(
    root: Any,
    *,
    run_path: str,
    payload: Mapping[str, Any],
    label: str,
) -> Any:
    if not isinstance(payload, Mapping):
        _fail(f"{label} payload is not an object.")
    node = _receipt_node(
        root,
        run_path=run_path,
        path=payload.get("array_ref"),
        label=label,
    )
    if not hasattr(node, "shape"):
        _fail(f"{label} does not resolve to an array.")
    try:
        expected_shape = tuple(int(value) for value in payload["shape"])
        expected_dtype = np.dtype(str(payload["dtype"]))
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"{label} has invalid shape/dtype metadata: {exc}.")
    if tuple(int(value) for value in node.shape) != expected_shape:
        _fail(
            f"{label} shape differs from its sealed receipt: "
            f"{tuple(node.shape)!r} != {expected_shape!r}."
        )
    try:
        actual_dtype = np.dtype(node.dtype)
    except (AttributeError, TypeError) as exc:
        _fail(f"{label} has no usable dtype: {exc}.")
    if actual_dtype != expected_dtype or actual_dtype.hasobject:
        _fail(
            f"{label} dtype differs from its sealed receipt: "
            f"{actual_dtype.str!r} != {expected_dtype.str!r}."
        )
    return node


def _receipt_record(
    root: Any,
    *,
    run_path: str,
    pointer: Mapping[str, Any],
    label: str,
) -> BoundCoordinateRecord:
    if not isinstance(pointer, Mapping) or set(pointer) != {
        "record_ref",
        "record_sha256",
    }:
        _fail(f"{label} is not an exact persisted-record pointer.")
    record_ref = pointer["record_ref"]
    if type(record_ref) is not str or record_ref.count("@") != 1:
        _fail(f"{label} has an invalid record reference.")
    node_path, attr_name = record_ref.split("@", 1)
    node = _receipt_node(
        root,
        run_path=run_path,
        path=node_path,
        label=f"{label} node",
    )
    try:
        bound = bind_persisted_coordinate_record(node, attr_name=attr_name)
    except Exception as exc:
        _fail(f"{label} is absent or invalid: {exc}.")
    if bound.record_ref != record_ref or bound.record_sha256 != pointer["record_sha256"]:
        _fail(f"{label} is stale.")
    return bound


def _receipt_live_attrs(
    node: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if _scientific_attrs(node, label=label) != dict(expected):
        _fail(f"{label} producer metadata changed after receipt sealing.")


def _receipt_validate_structure_tree(
    node: Any,
    expected: Mapping[str, Any],
    *,
    label: str,
    run_path: str,
) -> None:
    if not isinstance(expected, Mapping):
        _fail(f"{label} structure evidence is malformed.")
    kind = expected.get("kind")
    if kind == "array":
        if not hasattr(node, "shape"):
            _fail(f"{label} is not an array.")
        payload = expected.get("payload")
        if not isinstance(payload, Mapping):
            _fail(f"{label} lacks array payload metadata.")
        try:
            shape = tuple(int(value) for value in payload["shape"])
            dtype = np.dtype(str(payload["dtype"]))
        except (KeyError, TypeError, ValueError) as exc:
            _fail(f"{label} has invalid array metadata: {exc}.")
        if tuple(int(value) for value in node.shape) != shape or np.dtype(node.dtype) != dtype:
            _fail(f"{label} shape or dtype changed after receipt sealing.")
        _receipt_live_attrs(node, expected.get("producer_attrs", {}), label=label)
        return
    if kind != "group" or hasattr(node, "shape"):
        _fail(f"{label} structure kind is invalid.")
    children = expected.get("children")
    if not isinstance(children, Mapping):
        _fail(f"{label} group children are malformed.")
    if set(_member_names(node)) != set(children):
        _fail(f"{label} namespace members changed after receipt sealing.")
    _receipt_live_attrs(node, expected.get("producer_attrs", {}), label=label)
    for name, child in children.items():
        _receipt_validate_structure_tree(
            _child(node, name, label=f"{label}/{name}"),
            child,
            label=f"{label}/{name}",
            run_path=run_path,
        )


def _receipt_validate_closed_world_structure(
    context: BoundRefinedSubjectMaskCoordinateContext,
    structure: Mapping[str, Any],
) -> None:
    """Validate closed-world names and metadata without reading values."""

    if structure.get("policy") != "closed_world_exact_component_and_relation_namespaces_v2":
        _fail("Persisted refined closed-world structure policy is unsupported.")
    run = context._run_group
    root_arrays = structure.get("run_root", {}).get("arrays")
    root_groups = structure.get("run_root", {}).get("groups")
    if not isinstance(root_arrays, Mapping) or not isinstance(root_groups, Mapping):
        _fail("Persisted refined closed-world root inventory is malformed.")
    if set(_member_names(run)) != set(root_arrays) | set(root_groups):
        _fail("Refined run root namespace members changed after receipt sealing.")
    for name, evidence in root_arrays.items():
        node = _child(run, name, label=f"refined root array {name}")
        if not hasattr(node, "shape"):
            _fail(f"Refined root member {name!r} changed from array to group.")
        _receipt_array_node(
            context._root,
            run_path=context.run_path,
            payload=evidence.get("payload", {}),
            label=f"refined root array {name}",
        )
        _receipt_live_attrs(
            node,
            evidence.get("producer_attrs", {}),
            label=f"refined root array {name}",
        )
    for name, evidence in root_groups.items():
        node = _child(run, name, label=f"refined root group {name}")
        if hasattr(node, "shape"):
            _fail(f"Refined root member {name!r} changed from group to array.")
        if tuple(_member_names(node)) != tuple(evidence.get("members", ())):
            _fail(f"Refined root group {name!r} members changed after receipt sealing.")
        _receipt_live_attrs(
            node,
            evidence.get("producer_attrs", {}),
            label=f"refined root group {name}",
        )

    components_parent = _optional_child(run, "components")
    components = structure.get("components")
    if not isinstance(components, Mapping):
        _fail("Persisted refined component structure is malformed.")
    if components_parent is None:
        if any(entry.get("status") == "present" for entry in components.values()):
            _fail("Refined component namespaces disappeared after receipt sealing.")
    else:
        _receipt_live_attrs(
            components_parent,
            structure.get("components_parent_attrs", {}),
            label="refined components parent",
        )
        if set(_member_names(components_parent)) != {
            name for name, entry in components.items() if entry.get("status") == "present"
        }:
            _fail("Refined component namespace names changed after receipt sealing.")
    for component, evidence in components.items():
        group = _component_group(run, component)
        if evidence.get("status") == "absent":
            if group is not None:
                _fail(f"Refined component {component!r} appeared after receipt sealing.")
            continue
        if group is None:
            _fail(f"Refined component {component!r} disappeared after receipt sealing.")
        _receipt_live_attrs(group, evidence.get("producer_attrs", {}), label=component)
        arrays = evidence.get("arrays", {})
        groups = evidence.get("groups", {})
        if set(_member_names(group)) != set(arrays) | set(groups):
            _fail(f"Refined component {component!r} namespace members changed.")
        for name, item in arrays.items():
            node = _child(group, name, label=f"{component}/{name}")
            if not hasattr(node, "shape"):
                _fail(f"Refined component array {component}/{name} changed kind.")
            _receipt_array_node(
                context._root,
                run_path=context.run_path,
                payload=item.get("payload", {}),
                label=f"{component}/{name}",
            )
            _receipt_live_attrs(node, item.get("attrs", {}), label=f"{component}/{name}")
        for name, item in groups.items():
            node = _child(group, name, label=f"{component}/{name}")
            if item.get("classification") == "known_typed_namespace":
                _receipt_validate_structure_tree(
                    node,
                    item.get("exact_contents", {}),
                    label=f"{component}/{name}",
                    run_path=context.run_path,
                )
            elif item.get("classification") == "explicit_non_geometry":
                if hasattr(node, "shape"):
                    _fail(f"Explicit non-geometry namespace {component}/{name} changed kind.")
                _receipt_live_attrs(node, item.get("attrs", {}), label=f"{component}/{name}")
                arrays = item.get("arrays", {})
                if set(_member_names(node)) != set(arrays):
                    _fail(f"Explicit non-geometry namespace {component}/{name} changed members.")
                for child_name, payload in arrays.items():
                    _receipt_array_node(
                        context._root,
                        run_path=context.run_path,
                        payload=payload,
                        label=f"{component}/{name}/{child_name}",
                    )
            else:
                _fail(f"Refined component group {component}/{name} classification is unsupported.")

    relations = _optional_child(run, "relations")
    relation_inventory = structure.get("relations")
    if not isinstance(relation_inventory, Mapping):
        _fail("Persisted refined relation structure is malformed.")
    if relations is None:
        if relation_inventory:
            _fail("Refined relation namespaces disappeared after receipt sealing.")
    else:
        _receipt_live_attrs(
            relations,
            structure.get("relations_parent_attrs", {}),
            label="refined relations parent",
        )
        if set(_member_names(relations)) != set(relation_inventory):
            _fail("Refined relation namespace names changed after receipt sealing.")
        for name, evidence in relation_inventory.items():
            _receipt_validate_structure_tree(
                _child(relations, name, label=f"relation {name}"),
                evidence.get("exact_contents", {}),
                label=f"relation {name}",
                run_path=context.run_path,
            )


def _receipt_validate_component_qc(
    context: BoundRefinedSubjectMaskCoordinateContext,
    record: BoundCoordinateRecord,
) -> None:
    value = record.record
    if value.get("row_identity") != _record_pointer(context.row_identity):
        _fail("Persisted refined QC inventory row identity differs from context.")
    if value.get("component_labels") != _record_pointer(context.component_labels):
        _fail("Persisted refined QC inventory labels differ from context.")
    components = value.get("components")
    if not isinstance(components, Mapping):
        _fail("Persisted refined QC inventory components are malformed.")
    parent = _optional_child(context._run_group, "components")
    if set(components) != set(context.labels):
        _fail("Persisted refined QC inventory component set is not closed.")
    for component in context.labels:
        evidence = components[component]
        group = _optional_child(parent, component) if parent is not None else None
        qc = _optional_child(group, "qc") if group is not None else None
        if evidence.get("status") == "absent":
            if qc is not None:
                _fail(f"QC namespace for {component!r} appeared after receipt sealing.")
            continue
        if qc is None or hasattr(qc, "shape"):
            _fail(f"QC namespace for {component!r} changed kind or disappeared.")
        if canonical_node_path(qc) != evidence.get("path"):
            _fail(f"QC path for {component!r} changed after receipt sealing.")
        _receipt_live_attrs(qc, evidence.get("attrs", {}), label=f"{component} QC")
        arrays = evidence.get("arrays")
        if not isinstance(arrays, Mapping) or set(_member_names(qc)) != set(arrays):
            _fail(f"QC array names for {component!r} changed after receipt sealing.")
        for name, payload in arrays.items():
            _receipt_array_node(
                context._root,
                run_path=context.run_path,
                payload=payload,
                label=f"{component} QC {name}",
            )


def _receipt_coordinate_specs(
    context: BoundRefinedSubjectMaskCoordinateContext,
    inventory: BoundCoordinateRecord,
    measurement_authority: BoundCoordinateRecord,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, BoundCoordinateRecord],
    dict[str, BoundCoordinateRecord],
    dict[str, dict[str, Any]],
]:
    """Rebuild descriptor and measurement specs from sealed records only."""

    record = inventory.record
    canonical = record.get("canonical_geometry")
    ragged = record.get("ragged_geometry")
    measurement_surfaces = measurement_authority.record.get("surfaces")
    if not isinstance(canonical, Mapping) or not isinstance(ragged, Mapping):
        _fail("Persisted refined surface inventory lacks sealed geometry maps.")
    if not isinstance(measurement_surfaces, Mapping):
        _fail("Persisted refined measurement authority lacks sealed surfaces.")
    required_nodes = {
        "masks_roi": _child(
            context._run_group,
            "masks_roi",
            label="authoritative refined dense masks",
        ),
        "mask_present": _child(
            _child(context._run_group, "metrics", label="refined metrics"),
            "mask_present",
            label="mask_present",
        ),
        "area_px": _child(
            _child(context._run_group, "metrics", label="refined metrics"),
            "area_px",
            label="area_px",
        ),
        "centroid_xy": _child(
            _child(context._run_group, "metrics", label="refined metrics"),
            "centroid_xy",
            label="centroid_xy",
        ),
        "centroid_valid": _child(
            _child(context._run_group, "metrics", label="refined metrics"),
            "centroid_valid",
            label="centroid_valid",
        ),
        "bbox_xyxy": _child(
            _child(context._run_group, "metrics", label="refined metrics"),
            "bbox_xyxy",
            label="bbox_xyxy",
        ),
        "bbox_valid": _child(
            _child(context._run_group, "metrics", label="refined metrics"),
            "bbox_valid",
            label="bbox_valid",
        ),
        "available_channels": _child(
            context._run_group,
            "available_channels",
            label="available_channels",
        ),
    }
    metrics = _child(context._run_group, "metrics", label="refined metrics")
    _require_exact_members(
        metrics,
        ("mask_present", "area_px", "centroid_xy", "centroid_valid", "bbox_xyxy", "bbox_valid"),
        label="refined run metrics",
    )
    mask_shape = tuple(int(value) for value in required_nodes["masks_roi"].shape)
    if len(mask_shape) != 4 or mask_shape[0] != context.row_identity.leading_dimension:
        _fail("Refined masks_roi shape differs from sealed row identity.")
    if mask_shape[1] != len(context.labels):
        _fail("Refined masks_roi channel count differs from sealed labels.")
    expected_shapes = {
        "masks_roi": mask_shape,
        "mask_present": (mask_shape[0], mask_shape[1]),
        "area_px": (mask_shape[0], mask_shape[1]),
        "centroid_xy": (mask_shape[0], mask_shape[1], 2),
        "centroid_valid": (mask_shape[0], mask_shape[1]),
        "bbox_xyxy": (mask_shape[0], mask_shape[1], 4),
        "bbox_valid": (mask_shape[0], mask_shape[1]),
        "available_channels": (mask_shape[1],),
    }
    expected_dtypes = {
        "masks_roi": np.dtype("uint8"),
        "mask_present": np.dtype("bool"),
        "area_px": np.dtype("float32"),
        "centroid_xy": np.dtype("float32"),
        "centroid_valid": np.dtype("bool"),
        "bbox_xyxy": np.dtype("float32"),
        "bbox_valid": np.dtype("bool"),
        "available_channels": np.dtype("bool"),
    }
    for name, node in required_nodes.items():
        _require_shape_dtype(
            node,
            shape=expected_shapes[name],
            dtype=expected_dtypes[name],
            label=name,
        )
    if context._run_group.attrs.get("bbox_xyxy_convention") != "pixel_edge_half_open":
        _fail("Refined bbox_xyxy convention changed after receipt sealing.")
    if context._run_group.attrs.get("bbox_xyxy_derivation") != "foreground_half_open_pixel_edges_xyxy_v1":
        _fail("Refined bbox_xyxy derivation changed after receipt sealing.")

    payloads: dict[str, dict[str, Any]] = {}
    interpretations: dict[str, BoundCoordinateRecord] = {}
    specs = _required_geometry_specs(context, {
        "masks_roi": required_nodes["masks_roi"],
        "centroid_xy": required_nodes["centroid_xy"],
        "centroid_valid": required_nodes["centroid_valid"],
        "bbox_xyxy": required_nodes["bbox_xyxy"],
        "bbox_valid": required_nodes["bbox_valid"],
    })
    for path, entry in canonical.items():
        if not isinstance(entry, Mapping):
            _fail(f"Persisted canonical geometry entry {path!r} is malformed.")
        payload = entry.get("payload")
        node = _receipt_array_node(
            context._root,
            run_path=context.run_path,
            payload=payload,
            label=f"refined canonical geometry {path}",
        )
        payloads[path] = copy.deepcopy(dict(payload))
        interpretation = _receipt_record(
            context._root,
            run_path=context.run_path,
            pointer=entry.get("interpretation"),
            label=f"refined interpretation {path}",
        )
        interpretation_record = interpretation.record
        if str(interpretation_record.get("array_path", "")).strip("/") != str(
            payload.get("array_ref", "")
        ).strip("/"):
            _fail(f"Refined interpretation {path!r} array path is stale.")
        interpretations[path] = interpretation
        if path in specs:
            continue
        if path.endswith("/geometry/ellipse_params"):
            geometry_type = "ellipse_cxcy_wh_angle"
            components = ("center_x", "center_y", "width", "height", "angle")
            units = ("px", "px", "px", "px", "deg")
            convention = "continuous"
        elif path.endswith("/sampled_contours/points_xy"):
            geometry_type = "points_xy"
            components = ("x", "y")
            units = ("px", "px")
            convention = "pixel_center"
        else:
            _fail(f"Sealed refined canonical geometry path {path!r} is unsupported.")
        frame = (
            context.continuous_frame if convention == "continuous" else context.pixel_center_frame
        )
        chain = (
            context.continuous_chain if convention == "continuous" else context.pixel_center_chain
        )
        validity = interpretation_record.get("validity")
        validity_node = None
        if validity is not None:
            validity_node = _receipt_array_node(
                context._root,
                run_path=context.run_path,
                payload=validity.get("payload"),
                label=f"refined {path} validity",
            )
            payloads[f"{path}@validity"] = copy.deepcopy(dict(validity["payload"]))
        container_evidence = interpretation_record.get("container_evidence")
        if not isinstance(container_evidence, Mapping):
            _fail(f"Refined interpretation {path!r} lacks container evidence.")
        container = _receipt_node(
            context._root,
            run_path=context.run_path,
            path=container_evidence.get("path"),
            label=f"refined {path} container",
        )
        spec = {
            "node": node,
            "validity_node": validity_node,
            "geometry_type": geometry_type,
            "components": components,
            "component_units": units,
            "pixel_convention": convention,
            "frame": frame,
            "chain": chain,
            "overlay": CANONICAL_OVERLAY_REQUIRES_TRANSFORM,
            "component": interpretation_record.get("component"),
            "container": container,
            "invalid_value_policy": (
                validity.get("false_value_policy") if isinstance(validity, Mapping) else None
            ),
        }
        if path.endswith("/sampled_contours/points_xy"):
            count_path = path.removesuffix("points_xy") + "source_point_count"
            count_entry = measurement_surfaces.get(count_path)
            if not isinstance(count_entry, Mapping):
                _fail(f"Sealed sampled contour {path!r} lacks source-point-count evidence.")
            count_node = _receipt_array_node(
                context._root,
                run_path=context.run_path,
                payload=count_entry.get("payload"),
                label=f"refined {count_path}",
            )
            spec["companion_nodes"] = {"source_point_count": count_node}
            payloads[f"{path}@companion:source_point_count"] = copy.deepcopy(
                dict(count_entry["payload"])
            )
        specs[path] = spec

    ragged_specs: dict[str, dict[str, Any]] = {}
    ragged_records: dict[str, BoundCoordinateRecord] = {}
    for path, entry in ragged.items():
        if not isinstance(entry, Mapping):
            _fail(f"Persisted ragged geometry entry {path!r} is malformed.")
        payload = entry.get("payload")
        node = _receipt_array_node(
            context._root,
            run_path=context.run_path,
            payload=payload,
            label=f"refined ragged geometry {path}",
        )
        mapping = entry.get("row_mapping")
        if not isinstance(mapping, Mapping):
            _fail(f"Persisted ragged geometry {path!r} row mapping is malformed.")
        ptr_payload = mapping.get("ptr")
        len_payload = mapping.get("len")
        ptr_node = _receipt_array_node(
            context._root,
            run_path=context.run_path,
            payload=ptr_payload,
            label=f"refined ragged geometry {path} ptr",
        )
        len_node = _receipt_array_node(
            context._root,
            run_path=context.run_path,
            payload=len_payload,
            label=f"refined ragged geometry {path} len",
        )
        interpretation = _receipt_record(
            context._root,
            run_path=context.run_path,
            pointer=entry.get("interpretation"),
            label=f"refined ragged interpretation {path}",
        )
        ragged_records[path] = interpretation
        container_evidence = interpretation.record.get("container_evidence")
        container = _receipt_node(
            context._root,
            run_path=context.run_path,
            path=container_evidence.get("path") if isinstance(container_evidence, Mapping) else None,
            label=f"refined ragged {path} container",
        )
        ragged_specs[path] = {
            "node": node,
            "ptr_node": ptr_node,
            "len_node": len_node,
            "component": interpretation.record.get("component"),
            "point_count": int(mapping.get("point_count", 0)),
            "container": container,
        }
        payloads[path] = copy.deepcopy(dict(payload))
        payloads[f"{path}@ptr"] = copy.deepcopy(dict(ptr_payload))
        payloads[f"{path}@len"] = copy.deepcopy(dict(len_payload))

    measurement_specs: dict[str, dict[str, Any]] = {}
    for path, entry in measurement_surfaces.items():
        if not isinstance(entry, Mapping):
            _fail(f"Persisted measurement surface {path!r} is malformed.")
        node = _receipt_array_node(
            context._root,
            run_path=context.run_path,
            payload=entry.get("payload"),
            label=f"refined measurement {path}",
        )
        validity_entry = entry.get("validity")
        validity_node = None
        validity_policy = None
        if validity_entry is not None:
            if not isinstance(validity_entry, Mapping):
                _fail(f"Persisted measurement validity {path!r} is malformed.")
            validity_node = _receipt_array_node(
                context._root,
                run_path=context.run_path,
                payload=validity_entry.get("payload"),
                label=f"refined measurement {path} validity",
            )
            validity_policy = validity_entry.get("policy")
        axes = tuple(str(value) for value in entry.get("axis_order", ()))
        measurement_specs[path] = {
            "node": node,
            "quantity": entry.get("quantity"),
            "units": entry.get("units"),
            "operation": entry.get("operation"),
            "axes": axes,
            "coordinate_input_paths": tuple(entry.get("coordinate_input_paths", ())),
            "measurement_input_paths": tuple(entry.get("measurement_input_paths", ())),
            "row_axis_name": "observation" if "observation" in axes else None,
            "collection_axis": "subject_component" in axes and "observation" in axes,
            "selected_collection_members": tuple(entry.get("selected_collection_members", ())),
            "semantic_kind": entry.get("semantic_kind"),
            "validity_node": validity_node,
            "validity_policy": validity_policy,
        }

    return (
        required_nodes,
        specs,
        ragged_specs,
        interpretations,
        ragged_records,
        measurement_specs,
    )


def _load_refined_coordinate_surfaces_from_receipt(
    context: BoundRefinedSubjectMaskCoordinateContext,
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    payload = receipt["payload"]
    inventory = _receipt_record(
        context._root,
        run_path=context.run_path,
        pointer=payload["coordinate_records"]["surface_inventory"],
        label="refined surface inventory",
    )
    component_qc_inventory = _receipt_record(
        context._root,
        run_path=context.run_path,
        pointer=payload["coordinate_records"]["component_qc_inventory"],
        label="refined component QC inventory",
    )
    measurement_authority = _receipt_record(
        context._root,
        run_path=context.run_path,
        pointer=payload["coordinate_records"]["measurement_authority"],
        label="refined measurement authority",
    )
    scientific_manifest = _receipt_record(
        context._root,
        run_path=context.run_path,
        pointer=payload["coordinate_records"]["scientific_manifest"],
        label="refined scientific manifest",
    )
    _receipt_validate_closed_world_structure(
        context,
        inventory.record["closed_world_structure"],
    )
    _receipt_validate_component_qc(context, component_qc_inventory)
    if inventory.record.get("row_identity") != _record_pointer(context.row_identity):
        _fail("Refined surface inventory row identity differs from context.")
    if measurement_authority.record.get("surface_inventory") != _record_pointer(inventory):
        _fail("Refined measurement authority is not bound to the sealed inventory.")
    if scientific_manifest.record.get("surface_inventory") != _record_pointer(inventory):
        _fail("Refined scientific manifest is not bound to the sealed inventory.")
    specs_result = _receipt_coordinate_specs(context, inventory, measurement_authority)
    required_nodes, specs, ragged_specs, interpretations, ragged_records, measurement_specs = specs_result
    if scientific_manifest.record.get("coordinate_descriptor_paths") != sorted(specs):
        _fail("Refined scientific manifest descriptor paths changed after sealing.")
    descriptor_paths = scientific_manifest.record.get("measurement_descriptor_paths")
    if not isinstance(descriptor_paths, Mapping) or set(descriptor_paths) != set(measurement_specs):
        _fail("Refined scientific manifest measurement paths changed after sealing.")
    for path, table_payload in scientific_manifest.record.get("table_arrays", {}).items():
        _receipt_array_node(
            context._root,
            run_path=context.run_path,
            payload=table_payload,
            label=f"refined scientific manifest table {path}",
        )
    descriptors = _bindings(
        context,
        specs,
        inventory,
        interpretations,
        ragged_records,
        load=True,
    )
    measurements = _measurement_bindings(
        context,
        measurement_specs,
        descriptors,
        measurement_authority,
        load=True,
    )
    if {
        path: value.record_sha256 for path, value in measurements.items()
    } != {
        path: pointer["record_sha256"] for path, pointer in descriptor_paths.items()
    }:
        _fail("Refined measurement descriptors differ from the sealed manifest.")
    return BoundRefinedSubjectMaskCoordinateSurfaces(
        descriptors=descriptors,
        context=context,
        inventory=inventory,
        component_qc_inventory=component_qc_inventory,
        measurement_authority=measurement_authority,
        measurements=measurements,
        scientific_manifest=scientific_manifest,
        interpretations=interpretations,
        ragged_geometry=ragged_records,
        _verification_seal=_BOUND_SURFACES_SEAL,
    )


def _load_refined_coordinate_receipt_authority(
    root: Any,
    run_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = _node(root, run_path, label="refined coordinate successor")
    try:
        authority = load_coordinate_successor_authority(
            run,
            expected_kind=REFINED_SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
            expected_successor_run_path=run_path,
        )
        receipt = load_subject_mask_coordinate_validation_receipt(
            run,
            expected_kind=REFINED_SUBJECT_MASK_COORDINATE_VALIDATION_KIND,
            expected_successor_run_path=run_path,
            expected_coordinate_record_names=_REFINED_COORDINATE_VALIDATION_RECORD_NAMES,
        )
    except (CoordinateSuccessorAuthorityError, SubjectMaskCoordinateValidationReceiptError) as exc:
        raise RefinedSubjectMaskCoordinatePublicationError(
            f"Present refined coordinate validation receipt or successor authority is invalid: {exc}"
        ) from exc
    authority_payload = authority["payload"]
    target_manifest = run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(target_manifest, Mapping):
        _fail("Refined coordinate successor target manifest is absent.")
    manifest_errors = validate_subject_mask_core_run_manifest(target_manifest)
    if manifest_errors:
        _fail(
            "Refined coordinate successor target manifest is invalid: "
            + "; ".join(manifest_errors)
        )
    target_payload = target_manifest.get("payload", {})
    if (
        target_payload.get("run_id") != run_path.split("/", 1)[1]
        or target_payload.get("stage_family") != "refined_subject_masks_runs"
        or target_payload.get("kind") != "refined_dense_core"
    ):
        _fail("Refined coordinate successor target manifest path or kind is stale.")
    if run.attrs.get("coordinate_contract") != "canonical_v2":
        _fail("Refined coordinate successor target is not marked canonical_v2.")
    authority_records = authority_payload.get("coordinate_records")
    receipt_records = receipt["payload"]["coordinate_records"]
    if set(receipt_records) != _REFINED_COORDINATE_VALIDATION_RECORD_NAMES:
        _fail("Refined coordinate receipt record set is not exact.")
    if set(authority_records) != _REFINED_COORDINATE_AUTHORITY_RECORD_NAMES:
        _fail("Refined coordinate successor authority record set is not exact.")
    for name in _REFINED_COORDINATE_VALIDATION_RECORD_NAMES:
        if authority_records[name] != receipt_records[name]:
            _fail(f"Refined authority and receipt disagree for record {name!r}.")
    receipt_pointer = authority_records["coordinate_validation_receipt"]
    if receipt_pointer["record_ref"].split("@", 1)[0].strip("/") != run_path:
        _fail("Refined coordinate authority receipt pointer leaves the successor run.")
    source = authority_payload["source"]
    receipt_source = receipt["payload"]["source"]
    for name in (
        "run_path",
        "core_manifest_payload_digest",
        "core_manifest_document_digest",
        "logical_content_digest",
    ):
        if receipt_source.get(name) != source.get(
            {
                "core_manifest_payload_digest": "manifest_payload_digest",
                "core_manifest_document_digest": "manifest_document_digest",
                "logical_content_digest": "logical_content_digest",
            }.get(name, name)
        ):
            _fail(f"Refined coordinate receipt source {name!r} differs from authority.")
    source_path = source["run_path"]
    source_run = _node(root, source_path, label="refined coordinate source run")
    source_manifest = source_run.attrs.get(SUBJECT_MASK_CORE_RUN_MANIFEST_ATTRIBUTE)
    if not isinstance(source_manifest, Mapping):
        _fail("Refined coordinate source manifest is absent.")
    if (
        source["manifest_payload_digest"] != source_manifest.get("payload_digest")
        or source["manifest_document_digest"] != canonical_json_sha256(source_manifest)
        or source["logical_content_digest"]
        != source_manifest.get("payload", {}).get("logical_content", {}).get("digest")
        or receipt["payload"]["source"]["run_path"] != source_path
    ):
        _fail("Refined coordinate source manifest or logical-content digest is stale.")
    live_source_validation = source_manifest.get("payload", {}).get("source", {}).get(
        "validation_receipt"
    )
    receipt_validation = receipt["payload"]["source_validation"]
    if not isinstance(live_source_validation, Mapping):
        _fail("Refined coordinate source validation receipt is absent.")
    for name in (
        "schema_id",
        "schema_version",
        "payload_digest",
        "document_sha256",
        "semantic_unit_count",
    ):
        if receipt_validation.get(name) != live_source_validation.get(name):
            _fail(f"Refined coordinate source validation field {name!r} is stale.")
    target_logical = target_payload.get("logical_content", {})
    if target_logical.get("digest") != source["logical_content_digest"]:
        _fail("Refined coordinate target manifest logical-content digest is stale.")
    target_source = target_payload.get("source")
    source_source = source_manifest.get("payload", {}).get("source", {})
    if not isinstance(target_source, Mapping) or not isinstance(source_source, Mapping):
        _fail("Refined coordinate target manifest source identity is stale.")
    target_validation = target_source.get("validation_receipt")
    source_validation = source_source.get("validation_receipt")
    if not isinstance(target_validation, Mapping) or not isinstance(source_validation, Mapping):
        _fail("Refined coordinate target manifest source validation is stale.")
    for name in (
        "schema_id",
        "schema_version",
        "payload_digest",
        "document_sha256",
        "semantic_unit_count",
    ):
        if target_validation.get(name) != source_validation.get(name):
            _fail("Refined coordinate target manifest source identity is stale.")
    source_authority = authority_payload["source_authority"]
    if source_authority.get("kind") != "inactive_subject_mask_bundle_v3_plus_raw_successor":
        _fail("Refined coordinate source-authority kind is not the expected bundle successor authority.")
    source_record = source_authority.get("record")
    if not isinstance(source_record, Mapping) or set(source_record) != {
        "bundle_manifest",
        "raw_successor_authority",
    }:
        _fail("Refined coordinate source-authority bundle structure is malformed.")
    bundle_manifest = source_record.get("bundle_manifest")
    if not isinstance(bundle_manifest, Mapping):
        _fail("Refined coordinate successor bundle authority is absent.")
    bundle_errors = validate_subject_mask_bundle_manifest(bundle_manifest)
    if bundle_errors:
        _fail(
            "Refined coordinate successor bundle authority is invalid: "
            + "; ".join(bundle_errors)
        )
    if (
        receipt["payload"]["bundle_authority"]["kind"]
        != "inactive_subject_mask_bundle_v3"
        or bundle_manifest.get("schema_id") != SUBJECT_MASK_BUNDLE_MANIFEST_SCHEMA_ID
        or bundle_manifest.get("schema_version") != 3
        or receipt["payload"]["bundle_authority"]["document_digest"]
        != canonical_json_sha256(bundle_manifest)
    ):
        _fail("Refined coordinate bundle authority digest is stale.")
    raw_authority = source_record["raw_successor_authority"]
    raw_errors = validate_coordinate_successor_authority(
        raw_authority,
        expected_kind=SUBJECT_MASK_COORDINATE_SUCCESSOR_KIND,
    )
    if raw_errors:
        _fail(
            "Refined coordinate raw successor authority is invalid: "
            + "; ".join(raw_errors)
        )
    bundle_members = bundle_manifest.get("payload", {}).get("members")
    raw_member = bundle_members.get("raw") if isinstance(bundle_members, Mapping) else None
    raw_source = raw_authority.get("payload", {}).get("source", {})
    expected_raw_source = {
        "family": raw_member.get("family") if isinstance(raw_member, Mapping) else None,
        "run_path": raw_member.get("run_path") if isinstance(raw_member, Mapping) else None,
        "manifest_schema_id": (
            raw_member.get("manifest_schema_id")
            if isinstance(raw_member, Mapping)
            else None
        ),
        "manifest_schema_version": (
            raw_member.get("manifest_schema_version")
            if isinstance(raw_member, Mapping)
            else None
        ),
        "manifest_payload_digest": (
            raw_member.get("manifest_payload_digest")
            if isinstance(raw_member, Mapping)
            else None
        ),
        "manifest_document_digest": (
            raw_member.get("manifest_document_digest")
            if isinstance(raw_member, Mapping)
            else None
        ),
        "logical_content_digest": (
            raw_member.get("logical_content_digest")
            if isinstance(raw_member, Mapping)
            else None
        ),
    }
    if not isinstance(raw_member, Mapping) or raw_source != expected_raw_source:
        _fail(
            "Refined coordinate raw successor source authority is not bound to "
            "the bundle raw member."
        )
    authority_equivalence = authority_payload["payload_equivalence"]
    equivalence = authority_equivalence.get("payload_file_equivalence")
    receipt_equivalence = receipt["payload"]["payload_equivalence"]
    if not isinstance(equivalence, Mapping):
        _fail("Refined coordinate payload-file equivalence evidence is absent.")
    if any(
        equivalence.get(name) != receipt_equivalence.get(name)
        for name in ("schema_id", "schema_version", "receipt_digest", "inventory_digest", "payload_file_count")
    ):
        _fail("Refined coordinate payload-file equivalence evidence is stale.")
    if (
        authority_equivalence.get("source_logical_content_digest")
        != source["logical_content_digest"]
    ):
        _fail("Refined coordinate payload equivalence source digest is stale.")
    return receipt, authority


def _load_refined_subject_mask_coordinate_surfaces_impl(
    root: Any,
    run_path: str,
    *,
    require_complete: bool,
    require_activation_receipt: bool | None = None,
    expected_selector_eligible: bool,
    expected_publication_owner: str | None = None,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    context = _load_refined_subject_mask_coordinate_context(
        root,
        run_path,
        require_complete=require_complete,
        require_activation_receipt=require_activation_receipt,
        expected_selector_eligible=expected_selector_eligible,
        expected_publication_owner=expected_publication_owner,
    )
    # A complete refined run must fail if the exact selected raw payload was
    # changed after refinement.  Context preparation intentionally avoids this
    # large scan; complete-reader preflight performs it exactly once.
    if require_complete:
        try:
            raw_loader = (
                load_persisted_subject_mask_coordinate_surfaces
                if context.context_record.record.get(
                    "source_selector_eligible", True
                )
                else load_persisted_ineligible_subject_mask_coordinate_surfaces
            )
            raw = raw_loader(root, context.source.run_path)
        except Exception as exc:
            raise RefinedSubjectMaskCoordinatePublicationError(
                f"Selected raw subject-mask payload is stale or invalid: {exc}"
            ) from exc
        if (
            _source_authority_record(
                raw.context,
                context._run_group,
                context.labels,
                assignment_keypoints=context.assignment_keypoint_authority,
            )
            != context.source_authority.record
        ):
            _fail("Selected raw subject-mask authority changed after refinement.")
        if SUBJECT_MASK_COORDINATE_VALIDATION_RECEIPT_ATTRIBUTE in context._run_group.attrs:
            for attr_name in (
                "derived_mask_caches_stale",
                "metrics_stale",
                "contours_stale",
            ):
                if context._run_group.attrs.get(attr_name) is not False:
                    _fail(
                        "Receipt-backed refined loading requires explicit fresh "
                        f"{attr_name}=False."
                    )
            receipt, authority = _load_refined_coordinate_receipt_authority(
                root,
                context.run_path,
            )
            return _load_refined_coordinate_surfaces_from_receipt(
                context,
                receipt,
                authority,
            )
    (
        _required,
        specs,
        ragged_specs,
        measurement_specs,
        payloads,
        structure_inventory,
    ) = _surface_evidence(context)
    interpretations = _bind_interpretations(
        context,
        specs,
        payloads,
        stamp=False,
    )
    ragged_records = _bind_ragged_records(
        context,
        ragged_specs,
        payloads,
        stamp=False,
    )
    component_qc_inventory = bind_persisted_coordinate_record(
        context._run_group,
        attr_name=REFINED_SUBJECT_MASK_COMPONENT_QC_INVENTORY_ATTR,
    )
    if component_qc_inventory.record != _component_qc_inventory_record(context):
        _fail("Persisted refined component QC inventory differs from live QC evidence.")
    inventory = bind_persisted_coordinate_record(
        context._run_group,
        attr_name=REFINED_SUBJECT_MASK_SURFACE_INVENTORY_ATTR,
    )
    if inventory.record != _inventory_record(
        context,
        specs,
        ragged_specs,
        payloads,
        interpretations,
        ragged_records,
        component_qc_inventory,
        structure_inventory,
        measurement_specs,
    ):
        _fail("Persisted refined surface inventory differs from live geometry.")
    descriptors = _bindings(
        context,
        specs,
        inventory,
        interpretations,
        ragged_records,
        load=True,
    )
    measurement_authority = bind_persisted_coordinate_record(
        context._run_group,
        attr_name=REFINED_SUBJECT_MASK_MEASUREMENT_AUTHORITY_ATTR,
    )
    if measurement_authority.record != _measurement_authority_record(
        context,
        inventory=inventory,
        specs=measurement_specs,
    ):
        _fail("Persisted refined measurement authority differs from live evidence.")
    measurements = _measurement_bindings(
        context,
        measurement_specs,
        descriptors,
        measurement_authority,
        load=True,
    )
    scientific_manifest = bind_persisted_coordinate_record(
        context._run_group,
        attr_name=REFINED_SUBJECT_MASK_SCIENTIFIC_MANIFEST_ATTR,
    )
    if scientific_manifest.record != _scientific_manifest_record(
        context,
        inventory=inventory,
        specs=specs,
        payloads=payloads,
        measurement_authority=measurement_authority,
        measurements=measurements,
    ):
        _fail(
            "Persisted refined scientific manifest differs from live sealed surfaces."
        )
    return BoundRefinedSubjectMaskCoordinateSurfaces(
        descriptors=descriptors,
        context=context,
        inventory=inventory,
        component_qc_inventory=component_qc_inventory,
        measurement_authority=measurement_authority,
        measurements=measurements,
        scientific_manifest=scientific_manifest,
        interpretations=interpretations,
        ragged_geometry=ragged_records,
        _verification_seal=_BOUND_SURFACES_SEAL,
    )


def _load_refined_subject_mask_coordinate_surfaces(
    root: Any,
    run_path: str,
    *,
    require_complete: bool,
    require_activation_receipt: bool | None = None,
    expected_selector_eligible: bool,
    expected_publication_owner: str | None = None,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    if not require_complete:
        return _load_refined_subject_mask_coordinate_surfaces_impl(
            root,
            run_path,
            require_complete=require_complete,
            require_activation_receipt=require_activation_receipt,
            expected_selector_eligible=expected_selector_eligible,
            expected_publication_owner=expected_publication_owner,
        )
    with _payload_cache_scope():
        return _load_refined_subject_mask_coordinate_surfaces_impl(
            root,
            run_path,
            require_complete=require_complete,
            require_activation_receipt=require_activation_receipt,
            expected_selector_eligible=expected_selector_eligible,
            expected_publication_owner=expected_publication_owner,
        )


@proof_verification_operation
def load_persisted_refined_subject_mask_coordinate_surfaces(
    root: Any,
    run_path: str,
    *,
    expected_publication_owner: str | None = None,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    """Strict future-normal loader; legacy refined runs fail closed."""

    return _load_refined_subject_mask_coordinate_surfaces(
        root,
        run_path,
        require_complete=True,
        expected_selector_eligible=True,
        expected_publication_owner=expected_publication_owner,
    )


@proof_verification_operation
def load_persisted_ineligible_refined_subject_mask_coordinate_surfaces(
    root: Any,
    run_path: str,
    *,
    expected_publication_owner: str | None = None,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    """Load complete refined-mask surfaces that remain selector-ineligible.

    This verifies the coordinate publication only.  Consumers must separately
    prove the exact immutable subject-mask bundle that authorizes the member.
    """

    return _load_refined_subject_mask_coordinate_surfaces(
        root,
        run_path,
        require_complete=True,
        require_activation_receipt=False,
        expected_selector_eligible=False,
        expected_publication_owner=expected_publication_owner,
    )


def require_bound_refined_subject_mask_coordinate_surfaces(
    value: BoundRefinedSubjectMaskCoordinateSurfaces,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    """Freshly revalidate a previously sealed refined coordinate dependency."""

    if (
        type(value) is not BoundRefinedSubjectMaskCoordinateSurfaces
        or value._seal is not _BOUND_SURFACES_SEAL
    ):
        _fail("A sealed refined subject-mask coordinate dependency is required.")
    current = load_persisted_refined_subject_mask_coordinate_surfaces(
        value.context._root,
        value.context.run_path,
        expected_publication_owner=value.context.publication_owner,
    )
    if current.inventory.record_sha256 != value.inventory.record_sha256:
        _fail("Bound refined subject-mask coordinate dependency changed.")
    return current


@proof_verification_operation
def _load_completed_ineligible_refined_subject_mask_coordinate_surfaces(
    root: Any,
    run_path: str,
    *,
    require_activation_receipt: bool = False,
    expected_publication_owner: str | None = None,
) -> BoundRefinedSubjectMaskCoordinateSurfaces:
    """Validate a completed editable draft before its activation receipt exists."""

    return _load_refined_subject_mask_coordinate_surfaces(
        root,
        run_path,
        require_complete=True,
        require_activation_receipt=require_activation_receipt,
        expected_selector_eligible=False,
        expected_publication_owner=expected_publication_owner,
    )


def _snapshot_value(
    snapshot: Mapping[str, tuple[bool, Any]],
    name: str,
) -> tuple[bool, Any]:
    value = snapshot.get(name)
    if not isinstance(value, tuple) or len(value) != 2 or type(value[0]) is not bool:
        _fail(f"Refined selector snapshot lacks exact {name!r} state.")
    return value


def _require_snapshot_unchanged(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    *,
    names: Sequence[str] = _ACTIVATION_BASELINE_ATTRS,
) -> None:
    for name in names:
        present, value = _snapshot_value(snapshot, name)
        if (name in parent.attrs) is not present or (
            present and parent.attrs.get(name) != value
        ):
            _fail(
                f"Refined activation observed concurrent selector mutation of {name!r}."
            )


def _attr_state(attrs: Mapping[str, Any], name: str) -> tuple[bool, Any]:
    return (name in attrs, copy.deepcopy(attrs.get(name)))


def _require_exact_parent_states(
    parent: Any,
    expected: Mapping[str, tuple[bool, Any]],
) -> None:
    for name, state in expected.items():
        if _attr_state(parent.attrs, name) != state:
            _fail(
                f"Refined activation observed concurrent parent mutation of {name!r}."
            )


def _base_generation(snapshot: Mapping[str, tuple[bool, Any]]) -> int:
    present, value = _snapshot_value(
        snapshot,
        REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
    )
    if not present:
        return 0
    if type(value) is not int or value < 0:
        _fail("Refined parent publication generation must be nonnegative integer.")
    return value


def _restore_owned_parent_state(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
    *,
    run_name: str,
    owner: str,
    attempt_owned_states: Mapping[str, tuple[bool, Any]],
) -> None:
    attrs = parent.attrs
    failures: list[str] = []
    del run_name, owner  # Ownership is represented by exact attempted attr states.
    for name in reversed(tuple(attempt_owned_states)):
        try:
            attempted = attempt_owned_states[name]
            present, value = _snapshot_value(snapshot, name)
            current = _attr_state(attrs, name)
            original = (present, value)
            if current == original:
                continue
            if current != attempted:
                failures.append(
                    f"{name}: preserved alien state instead of restoring over it"
                )
                continue
            if present:
                attrs[name] = copy.deepcopy(value)
            elif name in attrs:
                del attrs[name]
        except BaseException as exc:  # pragma: no cover - hostile store
            failures.append(f"{name}: {exc}")
    if failures:
        raise RuntimeError(f"Refined selector rollback was incomplete: {failures!r}.")


@proof_verification_operation
def _activate_validated_refined_subject_mask_coordinate_surfaces(
    root: Any,
    run_parent: Any,
    value: BoundRefinedSubjectMaskCoordinateSurfaces,
    *,
    run_name: str,
    publication_owner_token: str,
    selector_snapshot: Mapping[str, tuple[bool, Any]],
) -> None:
    """Activate one fresh proof; eligibility is the literal final mutation."""

    if (
        type(value) is not BoundRefinedSubjectMaskCoordinateSurfaces
        or value._seal is not _BOUND_SURFACES_SEAL
    ):
        _fail("Refined activation requires sealed coordinate surfaces.")
    expected_path = f"refined_subject_masks_runs/{run_name}"
    context = value.context
    if (
        context.run_path != expected_path
        or context.selector_eligible is not False
        or context.completion_status not in (RUN_STATUS_RUNNING, RUN_STATUS_COMPLETE)
        or canonical_node_path(run_parent) != "refined_subject_masks_runs"
        or archive_identity(root) != archive_identity(run_parent)
    ):
        _fail("Refined activation proof does not name the exact ineligible child.")
    _publication_owner(context._run_group, expected=publication_owner_token)
    if (
        refined_subject_mask_lifecycle_state(context._run_group)
        != REFINED_SUBJECT_MASK_EDITABLE_DRAFT
    ):
        _fail("Refined activation requires one explicit editable draft.")

    def fresh_parent() -> Any:
        parent = _node(root, "refined_subject_masks_runs", label="refined parent")
        if archive_identity(parent) != archive_identity(run_parent):
            _fail("Refined parent changed archives during activation.")
        return parent

    attempt_owned_states: dict[str, tuple[bool, Any]] = {
        "latest_pending": (True, str(run_name)),
    }
    expected_parent_states = {
        name: _snapshot_value(selector_snapshot, name)
        for name in _ACTIVATION_BASELINE_ATTRS
    }
    # The snapshot may contain an older pending run (an activation attempt has
    # already installed this child) or this same durable editable draft. Exact
    # live-state validation below still requires latest_pending == run_name;
    # rollback restores whichever prior value the caller captured.
    _snapshot_value(selector_snapshot, "latest_pending")
    expected_parent_states["latest_pending"] = (True, str(run_name))

    def write_attr(parent: Any, name: str, value: Any) -> Any:
        attempted = (True, copy.deepcopy(value))
        attempt_owned_states[name] = attempted
        parent.attrs[name] = copy.deepcopy(value)
        expected_parent_states[name] = attempted
        current_parent = fresh_parent()
        _require_exact_parent_states(current_parent, expected_parent_states)
        return current_parent

    def delete_attr(parent: Any, name: str) -> Any:
        attempted = (False, None)
        attempt_owned_states[name] = attempted
        if name not in parent.attrs:
            _fail(f"Refined activation cannot delete missing attempt-owned {name!r}.")
        del parent.attrs[name]
        expected_parent_states[name] = attempted
        current_parent = fresh_parent()
        _require_exact_parent_states(current_parent, expected_parent_states)
        return current_parent

    try:
        parent = fresh_parent()
        _require_exact_parent_states(parent, expected_parent_states)
        _stamp_refined_subject_mask_activation_receipt(
            root,
            expected_path,
            owner=publication_owner_token,
        )
        parent = fresh_parent()
        _require_exact_parent_states(parent, expected_parent_states)
        current = _load_completed_ineligible_refined_subject_mask_coordinate_surfaces(
            root,
            expected_path,
            require_activation_receipt=True,
            expected_publication_owner=publication_owner_token,
        )
        if current.inventory.record_sha256 != value.inventory.record_sha256:
            _fail("Refined coordinate publication changed before activation.")
        parent = fresh_parent()
        _require_exact_parent_states(parent, expected_parent_states)
        # Close the completed-child proof phase before the publication lease or
        # any selector is mutated. Reuse accelerates validation but never
        # authorizes promotion from stale evidence.
        finish_proof_verification()
        base = _base_generation(selector_snapshot)
        policy_present, policy = _snapshot_value(
            selector_snapshot,
            REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
        )
        if policy_present and policy != _PUBLICATION_POLICY:
            _fail("Refined parent uses an unsupported publication policy.")
        lease_present, old_lease = _snapshot_value(
            selector_snapshot,
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
        )
        if lease_present:
            if (
                not isinstance(old_lease, dict)
                or old_lease.get("policy") != _PUBLICATION_POLICY
                or old_lease.get("next_generation") != base
                or old_lease.get("base_generation") != base - 1
            ):
                _fail(
                    "Refined parent already has an active or invalid publication lease."
                )
        lease = {
            "schema_id": "palette.refined_subject_mask_publication_lease",
            "schema_version": 1,
            "policy": _PUBLICATION_POLICY,
            "run_path": expected_path,
            "publication_owner": publication_owner_token,
            "base_generation": base,
            "next_generation": base + 1,
        }
        parent = write_attr(
            parent,
            REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR,
            lease,
        )
        parent = write_attr(parent, "latest_complete", run_name)
        parent = write_attr(parent, "latest", run_name)
        parent = delete_attr(parent, "latest_pending")
        parent = write_attr(
            parent,
            REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR,
            _PUBLICATION_POLICY,
        )
        parent = write_attr(
            parent,
            REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR,
            base + 1,
        )
        child = _fresh_owned_ineligible_run(
            root,
            expected_path,
            owner=publication_owner_token,
            statuses=(RUN_STATUS_COMPLETE,),
            label="Refined activation target",
        )
        stamp_refined_subject_mask_sealed_snapshot(child)
        # Commit point: there must be no fallible store operation after this
        # selector-eligibility write.  Readers independently revalidate every
        # parent selector, receipt, payload, and descriptor before use.
        child.attrs["stage_selector_eligible"] = True
    except BaseException as exc:
        try:
            child = _node(root, expected_path, label="refined activation child")
            committed = (
                child.attrs.get(REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR)
                == publication_owner_token
                and child.attrs.get("stage_selector_eligible") is True
            )
        except BaseException:
            committed = False
        if not committed:
            try:
                _restore_owned_parent_state(
                    fresh_parent(),
                    selector_snapshot,
                    run_name=run_name,
                    owner=publication_owner_token,
                    attempt_owned_states=attempt_owned_states,
                )
                child = _fresh_owned_ineligible_run(
                    root,
                    expected_path,
                    owner=publication_owner_token,
                    statuses=(RUN_STATUS_COMPLETE,),
                    label="Refined activation rollback target",
                )
                stamp_refined_subject_mask_editable_draft(child)
            except BaseException as rollback_exc:  # pragma: no cover - hostile store
                raise RefinedSubjectMaskCoordinatePublicationError(
                    "Refined activation failed and rollback was incomplete: "
                    f"{rollback_exc}."
                ) from exc
        raise


__all__ = [
    "REFINED_SUBJECT_MASK_ACTIVATION_RECEIPT_ATTR",
    "REFINED_SUBJECT_MASK_ASSIGNMENT_KEYPOINT_AUTHORITY_ATTR",
    "REFINED_SUBJECT_MASK_ARRAY_INTERPRETATION_ATTR",
    "REFINED_SUBJECT_MASK_COMPONENT_LABELS_ATTR",
    "REFINED_SUBJECT_MASK_COMPONENT_QC_INVENTORY_ATTR",
    "REFINED_SUBJECT_MASK_COORDINATE_CONTEXT_ATTR",
    "REFINED_SUBJECT_MASK_MEASUREMENT_AUTHORITY_ATTR",
    "REFINED_SUBJECT_MASK_PARENT_PUBLICATION_LEASE_ATTR",
    "REFINED_SUBJECT_MASK_PUBLICATION_GENERATION_ATTR",
    "REFINED_SUBJECT_MASK_PUBLICATION_OWNER_ATTR",
    "REFINED_SUBJECT_MASK_PUBLICATION_POLICY_ATTR",
    "REFINED_SUBJECT_MASK_RAGGED_GEOMETRY_ATTR",
    "REFINED_SUBJECT_MASK_SCIENTIFIC_MANIFEST_ATTR",
    "REFINED_SUBJECT_MASK_REFINEMENT_AUTHORITY_ATTR",
    "REFINED_SUBJECT_MASK_ROI_REFERENCE_EXTENT_ATTR",
    "REFINED_SUBJECT_MASK_SOURCE_AUTHORITY_ATTR",
    "REFINED_SUBJECT_MASK_SURFACE_INVENTORY_ATTR",
    "BoundRefinedSubjectMaskCoordinateContext",
    "BoundRefinedSubjectMaskCoordinateSurfaces",
    "RefinedSubjectMaskCoordinatePublicationCheckpoint",
    "RefinedSubjectMaskCoordinatePublicationError",
    "_activate_validated_refined_subject_mask_coordinate_surfaces",
    "_load_completed_ineligible_refined_subject_mask_coordinate_surfaces",
    "capture_refined_subject_mask_coordinate_publication_checkpoint",
    "load_persisted_refined_subject_mask_coordinate_surfaces",
    "load_persisted_ineligible_refined_subject_mask_coordinate_surfaces",
    "prepare_refined_subject_mask_coordinate_context",
    "publish_refined_subject_mask_coordinate_surfaces",
    "require_bound_refined_subject_mask_coordinate_surfaces",
    "rollback_refined_subject_mask_coordinate_publication",
]
