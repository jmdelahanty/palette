"""Sealed geometry-only crop profile validation and successor evidence.

This module retains the immutable v1 evidence vocabulary already stored on
published keypoint and subject-mask successors.  It no longer installs loader
adapters or mutates module globals.  Ordinary consumers resolve materialized
and sealed geometry-only profiles through the shared crop-source resolver;
the helpers here provide the full-strength geometry-only profile branch and
validate durable successor evidence written before that resolver existed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from fisheye.shared.archive_identity import archive_identity
from fisheye.shared.coordinate_surface_contract import (
    SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
)
from fisheye.shared.directed_transform_chain import (
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    load_bound_directed_transform_v2,
    stamp_directed_transform_v2,
)
from fisheye.shared.model_input_transform import ModelInputTransform
from fisheye.shared.pixel_frame_authority import (
    CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR,
    CROP_PLACEMENT_PADDED_PROVENANCE_ATTR,
    CROP_PLACEMENT_PADDED_PROVENANCE_DIGEST_ATTR,
    CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_ID,
    CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_VERSION,
    array_values_sha256,
    load_persisted_acquisition_camera_authority,
    load_normalized_pixel_frame_authority,
    load_source_camera_pixel_frame_authority,
    normalized_to_pixel_matrix,
    stamp_normalized_pixel_frame_authority,
)
from fisheye.shared.transform_authority import (
    load_bound_transform_authority,
    stamp_normalized_to_pixel_transform_authority,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_GEOMETRY_SCHEMA_V1,
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    CROP_RUN_MANIFEST_ATTRIBUTE,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.crop_schema import (
    CropPaddingMode,
    CropPlacementMode,
    CropSizeMode,
    crop_geometry_policy_from_manifest,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.coordinate_reference import canonical_node_path
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    RUN_STATUS_RUNNING,
)


# These string values are frozen in already-published v1 successor evidence.
# Neutral Python names prevent that compatibility vocabulary from defining the
# active architecture; changing the stored strings would invalidate artifacts.
SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_SCHEMA_ID = (
    "palette.coordinate_successor.historical_geometry_only_crop_adapter"
)
SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_SCHEMA_VERSION = 1
SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_KIND = (
    "sealed_geometry_only_crop_manifest_v2_no_pixel_source"
)
SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR = (
    "coordinate_successor_historical_bbox_normalization"
)
SEALED_GEOMETRY_BBOX_NORMALIZATION_SCHEMA_ID = (
    "palette.coordinate_successor.historical_bbox_normalization"
)
SEALED_GEOMETRY_BBOX_NORMALIZATION_SCHEMA_VERSION = 1
_BOUND_SEALED_GEOMETRY_CROP_SOURCE_SEAL = object()


class SealedGeometryCropProfileError(ValueError):
    """Raised when sealed geometry-crop profile validation fails."""


@dataclass(frozen=True, init=False)
class BoundSealedGeometryCropSource:
    """Validated geometry-only crop source exposed by the shared resolver."""

    crop_geometry: Any
    roi_geometry: Any
    roi_frame: Any
    bbox_roi_frame: Any
    crop_path: str
    crop_profile: str
    placement_ownership_attr: str
    placement_pixel_center_ownership_attr: str
    placement_pixel_edge_ownership_attr: str
    _root: Any
    _rowset_node: Any
    _placement_node: Any
    _roi_images_node: None
    _seal: object

    def __init__(self, *, _verification_seal: object | None = None, **values: Any) -> None:
        if _verification_seal is not _BOUND_SEALED_GEOMETRY_CROP_SOURCE_SEAL:
            raise SealedGeometryCropProfileError(
                "Sealed geometry crop sources must come from the validated profile resolver."
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _verification_seal)


@dataclass(frozen=True)
class SealedGeometryCropSuccessorBinding:
    """Exact sealed-crop and successor evidence, without loader replacement."""

    source: Any
    crop_path: str
    manifest: Mapping[str, Any]
    manifest_digest: str
    manifest_payload_digest: str
    logical_content_digest: str
    row_signatures_digest: str
    coordinate_catalog_digest: str
    n_frames: int
    n_instances: int
    source_width: int
    source_height: int
    source_run_path: str
    padded_lineage: Mapping[str, Any]
    successor_evidence_record: Mapping[str, Any]

    def as_record(self) -> dict[str, Any]:
        return dict(self.successor_evidence_record)


def _require_or_create_group(parent: Any, name: str) -> Any:
    try:
        child = parent[name]
    except (KeyError, TypeError):
        try:
            child = parent.create_group(name)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SealedGeometryCropProfileError(
                f"Cannot create successor coordinate-evidence group {name!r}: {exc}"
            ) from exc
    if not hasattr(child, "attrs") or not callable(
        getattr(child, "create_group", None)
    ):
        raise SealedGeometryCropProfileError(
            f"Successor coordinate-evidence child {name!r} is not a group."
        )
    return child


def _require_or_create_matrix(parent: Any, name: str, expected: np.ndarray) -> Any:
    matrix = np.asarray(expected)
    try:
        child = parent[name]
    except (KeyError, TypeError):
        try:
            child = parent.create_array(name, data=matrix, chunks=matrix.shape)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SealedGeometryCropProfileError(
                f"Cannot create successor coordinate-evidence matrix {name!r}: {exc}"
            ) from exc
    actual = _array(child, label=name)
    if (
        actual.dtype != matrix.dtype
        or actual.shape != matrix.shape
        or not np.array_equal(actual, matrix)
    ):
        raise SealedGeometryCropProfileError(
            f"Successor coordinate-evidence matrix {name!r} differs from the exact "
            "bbox-normalization formula."
        )
    return child


def bind_sealed_geometry_bbox_normalization_to_successor(
    binding: SealedGeometryCropSuccessorBinding,
    *,
    root: Any,
    successor_run: Any,
    successor_run_path: str,
) -> SealedGeometryCropSuccessorBinding:
    """Persist and bind the missing bbox-normalized authority on the successor.

    The sealed geometry-only crop contains source-camera point and bbox frames, but
    its geometry-only publication predates a reusable normalized-bbox endpoint.
    A coordinate successor therefore owns this small piece of derived evidence.
    It is never written back to the sealed geometry-only crop and is bound to that crop's
    exact frozen successor-evidence record and bbox camera authority.
    """

    if type(binding) is not SealedGeometryCropSuccessorBinding:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization requires one exact sealed crop binding."
        )
    path = str(successor_run_path).strip("/")
    if not path or path.count("/") != 1:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization requires one exact successor run path."
        )
    try:
        if archive_identity(root) != archive_identity(binding.source._root):
            raise SealedGeometryCropProfileError(
                "Sealed geometry-only bbox normalization target belongs to another archive."
            )
        crop = root[binding.crop_path]
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SealedGeometryCropProfileError):
            raise
        raise SealedGeometryCropProfileError(
            f"Sealed geometry-only bbox normalization target cannot be resolved: {exc}"
        ) from exc
    run = successor_run
    if canonical_node_path(run) != path:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization target node differs from its exact run path."
        )
    try:
        if archive_identity(run) != archive_identity(root):
            raise SealedGeometryCropProfileError(
                "Sealed geometry-only bbox normalization target node belongs to another archive."
            )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, SealedGeometryCropProfileError):
            raise
        raise SealedGeometryCropProfileError(
            f"Sealed geometry-only bbox normalization target identity is unavailable: {exc}"
        ) from exc
    attrs = getattr(run, "attrs", None)
    if not isinstance(attrs, Mapping) or not hasattr(attrs, "__setitem__"):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization target lacks writable attributes."
        )
    lifecycle_status = attrs.get(RUN_COMPLETION_STATUS_ATTR, attrs.get("status"))
    if (
        lifecycle_status != RUN_STATUS_RUNNING
        or attrs.get("stage_selector_eligible") is not False
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization requires a running selector-ineligible successor."
        )

    frames = _require_or_create_group(run, "coordinate_frames")
    frame_node = _require_or_create_group(
        frames, "historical_source_camera_normalized_bbox"
    )
    transforms = _require_or_create_group(run, "coordinate_transforms")
    bbox_camera = binding.source.crop_geometry.source_geometry.frame_evidence.bbox_source_camera_frame
    matrix_node = _require_or_create_matrix(
        transforms,
        "historical_source_camera_normalized_bbox_to_image",
        normalized_to_pixel_matrix(bbox_camera),
    )
    authority_node = _require_or_create_group(
        transforms,
        "historical_source_camera_normalized_bbox_to_image_authority",
    )
    token = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
    normalized_frame = stamp_normalized_pixel_frame_authority(
        frame_node,
        frame_id=f"historical_source_camera_normalized_bbox_{token}",
        pixel_frame=bbox_camera,
    )
    authority = stamp_normalized_to_pixel_transform_authority(
        authority_node,
        authority_id=f"historical_source_camera_normalized_bbox_to_image_{token}",
        matrix_node=matrix_node,
        source_frame=normalized_frame,
        target_frame=bbox_camera,
    )
    link = stamp_directed_transform_v2(
        matrix_node,
        transform_id=f"historical_source_camera_normalized_bbox_to_image_{token}",
        authority=authority,
        source_frame=normalized_frame,
        target_frame=bbox_camera,
    )
    chain = resolve_bound_directed_transform_chain((link,))

    record = {
        "schema_id": SEALED_GEOMETRY_BBOX_NORMALIZATION_SCHEMA_ID,
        "schema_version": SEALED_GEOMETRY_BBOX_NORMALIZATION_SCHEMA_VERSION,
        "successor_run_path": path,
        "historical_crop_adapter_sha256": canonical_json_sha256(binding.as_record()),
        "bbox_source_camera_frame": {
            "record_ref": bbox_camera.record_ref,
            "record_sha256": bbox_camera.record_sha256,
        },
        "normalized_frame": {
            "record_ref": normalized_frame.record_ref,
            "record_sha256": normalized_frame.record_sha256,
        },
        "normalized_to_source_camera": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in chain.transform_records
        ],
        "formula": "normalized_xy=image_xy/[reference_width_px,reference_height_px]",
        "reference_extent": {
            "width": int(bbox_camera.endpoint.width),
            "height": int(bbox_camera.endpoint.height),
            "units": "px",
        },
    }
    digest = canonical_json_sha256(record)
    existing = attrs.get(SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR)
    existing_digest = attrs.get(f"{SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR}_sha256")
    if existing is not None or existing_digest is not None:
        if existing != record or existing_digest != digest:
            raise SealedGeometryCropProfileError(
                "Successor frozen-v1 bbox-normalization evidence already exists "
                "with different content."
            )
    else:
        attrs[SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR] = copy.deepcopy(record)
        attrs[f"{SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR}_sha256"] = digest

    frame_evidence = copy.copy(
        binding.source.crop_geometry.source_geometry.frame_evidence
    )
    frame_evidence.normalized_frame = normalized_frame
    frame_evidence.normalized_to_source_camera = chain
    source_geometry = copy.copy(binding.source.crop_geometry.source_geometry)
    source_geometry.frame_evidence = frame_evidence
    crop_geometry = copy.copy(binding.source.crop_geometry)
    crop_geometry.source_geometry = source_geometry
    source = BoundSealedGeometryCropSource(
        crop_geometry=crop_geometry,
        roi_geometry=binding.source.roi_geometry,
        roi_frame=binding.source.roi_frame,
        bbox_roi_frame=binding.source.bbox_roi_frame,
        crop_path=binding.source.crop_path,
        crop_profile=binding.source.crop_profile,
        placement_ownership_attr=binding.source.placement_ownership_attr,
        placement_pixel_center_ownership_attr=(
            binding.source.placement_pixel_center_ownership_attr
        ),
        placement_pixel_edge_ownership_attr=(
            binding.source.placement_pixel_edge_ownership_attr
        ),
        _root=root,
        _rowset_node=crop,
        _placement_node=crop["source_crop_xywh"],
        _roi_images_node=None,
        _verification_seal=_BOUND_SEALED_GEOMETRY_CROP_SOURCE_SEAL,
    )
    return replace(binding, source=source)


def load_sealed_geometry_bbox_normalization_from_successor(
    binding: SealedGeometryCropSuccessorBinding,
    *,
    root: Any,
    successor_run: Any,
    successor_run_path: str,
) -> SealedGeometryCropSuccessorBinding:
    """Reload the successor-owned normalized-bbox endpoint fail closed."""

    if type(binding) is not SealedGeometryCropSuccessorBinding:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization requires one exact sealed crop binding."
        )
    path = str(successor_run_path).strip("/")
    if not path or path.count("/") != 1 or canonical_node_path(successor_run) != path:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization requires one exact successor run path."
        )
    if (
        archive_identity(root) != archive_identity(binding.source._root)
        or archive_identity(successor_run) != archive_identity(root)
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization target belongs to another archive."
        )
    attrs = getattr(successor_run, "attrs", None)
    lifecycle_status = (
        attrs.get(RUN_COMPLETION_STATUS_ATTR, attrs.get("status"))
        if isinstance(attrs, Mapping)
        else None
    )
    if lifecycle_status != "complete":
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only bbox normalization reload requires a complete successor."
        )
    bind_persisted_run_attribute_record(
        successor_run,
        attr_name=SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR,
    )
    record = attrs[SEALED_GEOMETRY_BBOX_NORMALIZATION_ATTR]

    bbox_camera = (
        binding.source.crop_geometry.source_geometry.frame_evidence
        .bbox_source_camera_frame
    )
    frame_node = successor_run["coordinate_frames"][
        "historical_source_camera_normalized_bbox"
    ]
    normalized_frame = load_normalized_pixel_frame_authority(
        frame_node,
        pixel_frame=bbox_camera,
    )
    transforms = successor_run["coordinate_transforms"]
    matrix_node = transforms[
        "historical_source_camera_normalized_bbox_to_image"
    ]
    authority_node = transforms[
        "historical_source_camera_normalized_bbox_to_image_authority"
    ]
    authority = load_bound_transform_authority(
        authority_node,
        payload_node=matrix_node,
        source_frame=normalized_frame,
        target_frame=bbox_camera,
    )
    link = load_bound_directed_transform_v2(
        matrix_node,
        authority=authority,
        source_frame=normalized_frame,
        target_frame=bbox_camera,
    )
    chain = resolve_bound_directed_transform_chain((link,))
    expected = {
        "schema_id": SEALED_GEOMETRY_BBOX_NORMALIZATION_SCHEMA_ID,
        "schema_version": SEALED_GEOMETRY_BBOX_NORMALIZATION_SCHEMA_VERSION,
        "successor_run_path": path,
        "historical_crop_adapter_sha256": canonical_json_sha256(
            binding.as_record()
        ),
        "bbox_source_camera_frame": {
            "record_ref": bbox_camera.record_ref,
            "record_sha256": bbox_camera.record_sha256,
        },
        "normalized_frame": {
            "record_ref": normalized_frame.record_ref,
            "record_sha256": normalized_frame.record_sha256,
        },
        "normalized_to_source_camera": [
            {
                "record_ref": item.record_ref,
                "record_sha256": item.record_sha256,
            }
            for item in chain.transform_records
        ],
        "formula": (
            "normalized_xy=image_xy/[reference_width_px,reference_height_px]"
        ),
        "reference_extent": {
            "width": int(bbox_camera.endpoint.width),
            "height": int(bbox_camera.endpoint.height),
            "units": "px",
        },
    }
    if record != expected:
        raise SealedGeometryCropProfileError(
            "Persisted frozen-v1 bbox-normalization evidence changed."
        )

    crop = root[binding.crop_path]
    frame_evidence = copy.copy(
        binding.source.crop_geometry.source_geometry.frame_evidence
    )
    frame_evidence.normalized_frame = normalized_frame
    frame_evidence.normalized_to_source_camera = chain
    source_geometry = copy.copy(binding.source.crop_geometry.source_geometry)
    source_geometry.frame_evidence = frame_evidence
    crop_geometry = copy.copy(binding.source.crop_geometry)
    crop_geometry.source_geometry = source_geometry
    source = BoundSealedGeometryCropSource(
        crop_geometry=crop_geometry,
        roi_geometry=binding.source.roi_geometry,
        roi_frame=binding.source.roi_frame,
        bbox_roi_frame=binding.source.bbox_roi_frame,
        crop_path=binding.source.crop_path,
        crop_profile=binding.source.crop_profile,
        placement_ownership_attr=binding.source.placement_ownership_attr,
        placement_pixel_center_ownership_attr=(
            binding.source.placement_pixel_center_ownership_attr
        ),
        placement_pixel_edge_ownership_attr=(
            binding.source.placement_pixel_edge_ownership_attr
        ),
        _root=root,
        _rowset_node=crop,
        _placement_node=crop["source_crop_xywh"],
        _roi_images_node=None,
        _verification_seal=_BOUND_SEALED_GEOMETRY_CROP_SOURCE_SEAL,
    )
    return replace(binding, source=source)


def _padded_lineage_summary(
    values: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    roi_width: int,
    roi_height: int,
    crop_policy_payload_digest: str,
    origin_authority_digest: str,
    provider_record_sha256: str,
) -> dict[str, Any]:
    """Freeze requested/clipped/padding semantics without storing pixel data."""

    placement = np.asarray(values)
    if (
        placement.dtype != np.dtype("<f4")
        or placement.ndim != 2
        or placement.shape[1] != 4
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only padded crop placement must be a little-endian float32 (N,4) array."
        )
    if placement.shape[0] == 0:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only padded crop placement must contain at least one row."
        )
    if not np.isfinite(placement).all():
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only padded crop placement must be finite."
        )
    numeric = placement.astype(np.float64, copy=False)
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only padded crop placement must contain exact integer-valued coordinates."
        )
    requested = rounded.astype(np.int64)
    if (
        np.any(requested[:, 2] != int(roi_width))
        or np.any(requested[:, 3] != int(roi_height))
        or np.any(requested[:, 2:] <= 0)
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only padded crop placement must use the exact fixed positive ROI extent."
        )
    clipped_x0 = np.maximum(requested[:, 0], 0)
    clipped_y0 = np.maximum(requested[:, 1], 0)
    clipped_x1 = np.minimum(requested[:, 0] + requested[:, 2], int(source_width))
    clipped_y1 = np.minimum(requested[:, 1] + requested[:, 3], int(source_height))
    if np.any(clipped_x1 <= clipped_x0) or np.any(clipped_y1 <= clipped_y0):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only padded crop placement contains a row with no source-camera intersection."
        )
    clipped = np.column_stack(
        (clipped_x0, clipped_y0, clipped_x1 - clipped_x0, clipped_y1 - clipped_y0)
    ).astype("<i8", copy=False)
    padding = np.column_stack(
        (
            np.maximum(0, -requested[:, 0]),
            np.maximum(0, -requested[:, 1]),
            np.maximum(0, requested[:, 0] + requested[:, 2] - int(source_width)),
            np.maximum(0, requested[:, 1] + requested[:, 3] - int(source_height)),
        )
    ).astype("<i8", copy=False)
    padded_rows = np.any(padding != 0, axis=1)
    provenance = {
        "schema_id": CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_ID,
        "schema_version": CROP_PLACEMENT_PADDED_PROVENANCE_SCHEMA_VERSION,
        "crop_policy_payload_digest": crop_policy_payload_digest,
        "origin_authority_digest": origin_authority_digest,
        "provider_record_sha256": provider_record_sha256,
    }
    return {
        "schema_id": "palette.coordinate_successor.padded_crop_lineage",
        "schema_version": 1,
        "requested_roi": {
            "array_role": "source_crop_xywh",
            "dtype": placement.dtype.str,
            "shape": [int(value) for value in placement.shape],
            "extent": {
                "width": int(roi_width),
                "height": int(roi_height),
                "units": "px",
            },
            "window_policy": "requested_window_zero_padded_v1",
        },
        "source_camera_extent": {
            "width": int(source_width),
            "height": int(source_height),
            "units": "px",
        },
        "clipped_source_camera_intersection": {
            "formula": "x0=max(requested_x,0); y0=max(requested_y,0); x1=min(requested_x+width,W); y1=min(requested_y+height,H)",
            "dtype": clipped.dtype.str,
            "shape": [int(value) for value in clipped.shape],
            "sha256": array_values_sha256(clipped),
        },
        "zero_padding_offsets_ltrb": {
            "formula": "left=max(0,-x); top=max(0,-y); right=max(0,x+width-W); bottom=max(0,y+height-H)",
            "dtype": padding.dtype.str,
            "shape": [int(value) for value in padding.shape],
            "sha256": array_values_sha256(padding),
        },
        "padded_row_count": int(np.count_nonzero(padded_rows)),
        "padded_row_fraction": float(np.mean(padded_rows))
        if placement.shape[0]
        else 0.0,
        "max_padding_ltrb": [int(value) for value in padding.max(axis=0, initial=0)],
        "crop_policy_provenance": provenance,
        "crop_local_to_source_camera": {
            "formula": "source_xy = requested_origin_xy + crop_local_xy",
            "source_pixel_authority": "clipped_source_camera_intersection_only",
            "source_pixels_outside_extent": "synthetic_zero_padding_no_source_pixel_correspondence",
        },
    }


def bind_persisted_padded_placement_record(
    run: Any,
    *,
    attr_name: str = CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
) -> dict[str, Any]:
    """Return a digest-bound pointer to a successor-owned padded placement attr."""

    try:
        placement = run["source_crop_xywh"]
    except (KeyError, TypeError) as exc:
        raise SealedGeometryCropProfileError(
            "Successor lacks its source_crop_xywh placement node."
        ) from exc
    raw = getattr(placement, "attrs", {}).get(attr_name)
    if not isinstance(raw, Mapping):
        raise SealedGeometryCropProfileError(
            f"Successor placement lacks durable padded ownership attr {attr_name!r}."
        )
    digest = canonical_json_sha256(raw)
    if getattr(placement, "attrs", {}).get(f"{attr_name}_sha256") != digest:
        raise SealedGeometryCropProfileError(
            f"Successor placement has a stale padded ownership digest for {attr_name!r}."
        )
    provenance = bind_persisted_run_attribute_record(
        placement,
        attr_name=CROP_PLACEMENT_PADDED_PROVENANCE_ATTR,
    )
    return {
        "record_ref": f"/{canonical_node_path(placement)}@{attr_name}",
        "record_sha256": digest,
        "provenance": provenance,
    }


def bind_persisted_run_attribute_record(
    node: Any,
    *,
    attr_name: str,
) -> dict[str, Any]:
    """Bind a pointer to one exact persisted node attribute."""

    attrs = getattr(node, "attrs", None)
    value = attrs.get(attr_name) if isinstance(attrs, Mapping) else None
    if not isinstance(value, Mapping):
        raise SealedGeometryCropProfileError(
            f"Node {canonical_node_path(node)!r} lacks durable mapping attr {attr_name!r}."
        )
    digest = canonical_json_sha256(value)
    stored_digest = attrs.get(f"{attr_name}_sha256")
    if stored_digest != digest:
        raise SealedGeometryCropProfileError(
            f"Node {canonical_node_path(node)!r} has a missing or stale digest for attr {attr_name!r}."
        )
    return {
        "record_ref": f"/{canonical_node_path(node)}@{attr_name}",
        "record_sha256": digest,
    }


def stamp_persisted_padded_placement_provenance(
    run: Any,
    binding: SealedGeometryCropSuccessorBinding,
) -> dict[str, Any]:
    """Persist the successor-owned proof needed by padded placement stamping."""

    placement = run["source_crop_xywh"]
    attrs = getattr(placement, "attrs", None)
    if not isinstance(attrs, Mapping) or not hasattr(attrs, "__setitem__"):
        raise SealedGeometryCropProfileError(
            "Successor placement does not expose writable attrs for padded provenance."
        )
    lineage = binding.padded_lineage
    provenance = dict(lineage["crop_policy_provenance"])
    if set(provenance) != {
        "schema_id",
        "schema_version",
        "crop_policy_payload_digest",
        "origin_authority_digest",
        "provider_record_sha256",
    }:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only padded lineage has an incomplete crop-policy provenance record."
        )
    digest = canonical_json_sha256(provenance)
    existing = attrs.get(CROP_PLACEMENT_PADDED_PROVENANCE_ATTR)
    existing_digest = attrs.get(CROP_PLACEMENT_PADDED_PROVENANCE_DIGEST_ATTR)
    if existing is not None or existing_digest is not None:
        if existing != provenance or existing_digest != digest:
            raise SealedGeometryCropProfileError(
                "Successor padded placement provenance already exists with different content."
            )
    else:
        attrs[CROP_PLACEMENT_PADDED_PROVENANCE_ATTR] = copy.deepcopy(provenance)
        attrs[CROP_PLACEMENT_PADDED_PROVENANCE_DIGEST_ATTR] = digest
    return bind_persisted_run_attribute_record(
        placement,
        attr_name=CROP_PLACEMENT_PADDED_PROVENANCE_ATTR,
    )


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        value = np.asarray(node[...])
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise SealedGeometryCropProfileError(
            f"Sealed geometry-only crop array {label!r} cannot be decoded: {exc}"
        ) from exc
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SealedGeometryCropProfileError(
            f"Sealed geometry-only crop {label} must be an object."
        )
    return value


def _require_crop_reference(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    crop_path: str,
) -> None:
    payload = _require_mapping(manifest.get("payload"), label="manifest payload")
    logical = _require_mapping(payload.get("logical_content"), label="logical content")
    coordinate = _require_mapping(
        payload.get("coordinate_contract"), label="coordinate contract"
    )
    expected = {
        "run_path": crop_path,
        "manifest_payload_digest": manifest.get("payload_digest"),
        "logical_content_digest": logical.get("digest"),
        "coordinate_catalog_digest": coordinate.get("digest"),
    }
    payload_run_id = payload.get("run_id")
    if value.get("run_id") is not None and value.get("run_id") != payload_run_id:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop reference run_id does not match the exact crop manifest."
        )
    for name, actual in expected.items():
        if name in value and value.get(name) != actual:
            raise SealedGeometryCropProfileError(
                f"Sealed geometry-only crop reference {name!r} does not match the exact crop manifest."
            )
    if "manifest_digest" in value and value.get(
        "manifest_digest"
    ) != canonical_json_sha256(manifest):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop reference manifest_digest does not match the exact crop manifest."
        )
    if "row_signatures_digest" in value:
        declared = (
            logical.get("document", {})
            .get("arrays", {})
            .get("source_row_signature", {})
        )
        if value.get("row_signatures_digest") != declared.get("sha256"):
            raise SealedGeometryCropProfileError(
                "Sealed geometry-only crop reference row-signatures digest differs from the exact crop manifest."
            )
    declared_dimensions = payload.get("logical_schema", {}).get("dimensions", {})
    reference_dimensions = value.get("dimensions")
    if reference_dimensions is not None:
        if not isinstance(reference_dimensions, Mapping) or any(
            reference_dimensions.get(name) != declared_dimensions.get(name)
            for name in ("n_frames", "n_instances", "source_width", "source_height")
        ):
            raise SealedGeometryCropProfileError(
                "Sealed geometry-only crop reference dimensions differ from the exact crop manifest."
            )


def _source_dimensions(source_manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    payload = _require_mapping(
        source_manifest.get("payload"), label="source manifest payload"
    )
    logical = _require_mapping(
        payload.get("logical_schema"), label="source logical schema"
    )
    dimensions = _require_mapping(logical.get("dimensions"), label="source dimensions")
    values: list[int] = []
    for name in ("n_frames", "n_instances"):
        value = dimensions.get(name)
        if type(value) is not int or value < 0:
            raise SealedGeometryCropProfileError(
                f"Source manifest dimension {name!r} is not an exact nonnegative integer."
            )
        values.append(value)
    return values[0], values[1], int(dimensions.get("n_keypoints", 0) or 0)


def _validate_source_selection(
    *,
    source_manifest: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
    crop_arrays: Mapping[str, Any],
    n_frames: int,
    n_instances: int,
    source_run_path: str,
) -> None:
    required = {
        "source_crop_row_ids",
        "instance_key",
        "source_acquisition_frame_index",
    }
    missing = sorted(required - set(source_arrays))
    if missing:
        raise SealedGeometryCropProfileError(
            f"{source_run_path} lacks required crop-selection arrays: {missing!r}."
        )
    rows = _array(source_arrays["source_crop_row_ids"], label="source_crop_row_ids")
    keys = _array(source_arrays["instance_key"], label="source instance_key")
    frames = _array(
        source_arrays["source_acquisition_frame_index"],
        label="source_acquisition_frame_index",
    )
    if rows.dtype != np.dtype("<i8") or rows.shape != (n_instances,):
        raise SealedGeometryCropProfileError(
            "Source crop row selection has the wrong dtype, shape, or cardinality."
        )
    if keys.dtype != np.dtype("<u8") or keys.shape != (n_instances,):
        raise SealedGeometryCropProfileError(
            "Source instance_key has the wrong dtype, shape, or cardinality."
        )
    if frames.dtype != np.dtype("<i8") or frames.shape != (n_instances,):
        raise SealedGeometryCropProfileError(
            "Source acquisition-frame array has the wrong dtype, shape, or cardinality."
        )
    expected_rows = np.arange(n_instances, dtype=np.int64)
    if not np.array_equal(np.sort(rows), expected_rows):
        raise SealedGeometryCropProfileError(
            "Source crop row selection is not an exact complete permutation of the crop rowset."
        )
    crop_keys = _array(crop_arrays["instance_key"], label="crop instance_key")
    crop_frames = _array(
        crop_arrays["source_acquisition_frame_index"],
        label="crop source_acquisition_frame_index",
    )
    if not np.array_equal(keys, crop_keys[rows]):
        raise SealedGeometryCropProfileError(
            "Source instance_key values do not match the exact selected crop rows."
        )
    if not np.array_equal(frames, crop_frames[rows]):
        raise SealedGeometryCropProfileError(
            "Source acquisition-frame values do not match the exact selected crop rows."
        )
    if "source_crop_row_signature" in source_arrays:
        source_signatures = _array(
            source_arrays["source_crop_row_signature"],
            label="source_crop_row_signature",
        )
        crop_signatures = _array(
            crop_arrays["source_row_signature"],
            label="crop source_row_signature",
        )
        if source_signatures.dtype != np.dtype("uint8") or source_signatures.shape != (
            n_instances,
            32,
        ):
            raise SealedGeometryCropProfileError(
                "Source crop row signatures have the wrong dtype or shape."
            )
        if not np.array_equal(source_signatures, crop_signatures[rows]):
            raise SealedGeometryCropProfileError(
                "Source crop row signatures do not match the exact crop rows."
            )
    if "source_crop_xywh" in source_arrays:
        source_placement = _array(
            source_arrays["source_crop_xywh"], label="source_crop_xywh"
        )
        crop_placement = _array(
            crop_arrays["source_crop_xywh"], label="crop source_crop_xywh"
        )
        if source_placement.dtype != np.dtype("float32") or source_placement.shape != (
            n_instances,
            4,
        ):
            raise SealedGeometryCropProfileError(
                "Source crop placement has the wrong dtype or shape."
            )
        if not np.array_equal(source_placement, crop_placement[rows]):
            raise SealedGeometryCropProfileError(
                "Source crop placement does not match the exact crop rows."
            )
    del source_manifest, n_frames


def _frame_endpoint(width: int, height: int) -> SimpleNamespace:
    return SimpleNamespace(width=width, height=height)


def _build_bound_source(
    *,
    root: Any,
    crop: Any,
    crop_path: str,
    manifest: Mapping[str, Any],
    n_instances: int,
    roi_width: int,
    roi_height: int,
    point_camera: Any,
    bbox_camera: Any,
    acquisition_frame: Any,
) -> BoundSealedGeometryCropSource:
    manifest_digest = canonical_json_sha256(manifest)
    manifest_record_ref = f"/{crop_path}@run_manifest"
    selection = SimpleNamespace(
        record_ref=manifest_record_ref,
        record_sha256=manifest_digest,
    )
    roi_derivation = SimpleNamespace(
        record_ref=manifest_record_ref,
        record_sha256=manifest_digest,
    )
    frame_record = SimpleNamespace(
        record_ref=manifest_record_ref,
        record_sha256=manifest_digest,
    )
    source_geometry = SimpleNamespace(
        frame_evidence=SimpleNamespace(
            source_camera_frame=point_camera,
            bbox_source_camera_frame=bbox_camera,
            acquisition_frame=acquisition_frame,
            normalized_frame=SimpleNamespace(
                record_ref=manifest_record_ref,
                record_sha256=manifest_digest,
            ),
            normalized_to_source_camera=SimpleNamespace(
                transform_records=(
                    SimpleNamespace(
                        record_ref=manifest_record_ref,
                        record_sha256=manifest_digest,
                    ),
                ),
            ),
        )
    )
    crop_geometry = SimpleNamespace(
        source_geometry=source_geometry,
        row_identity=SimpleNamespace(
            leading_dimension=n_instances,
            record_ref=frame_record.record_ref,
            record_sha256=frame_record.record_sha256,
        ),
        selection_derivation=selection,
    )
    roi_frame = SimpleNamespace(
        endpoint=_frame_endpoint(roi_width, roi_height),
        record_ref=frame_record.record_ref,
        record_sha256=frame_record.record_sha256,
        pixel_convention=SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
    )
    bbox_roi_frame = SimpleNamespace(
        endpoint=_frame_endpoint(roi_width, roi_height),
        record_ref=frame_record.record_ref,
        record_sha256=frame_record.record_sha256,
        pixel_convention=SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    )
    return BoundSealedGeometryCropSource(
        crop_geometry=crop_geometry,
        roi_geometry=SimpleNamespace(derivation=roi_derivation),
        roi_frame=roi_frame,
        bbox_roi_frame=bbox_roi_frame,
        crop_path=crop_path,
        crop_profile="sealed_geometry_only_v2",
        placement_ownership_attr=CROP_PLACEMENT_PADDED_OWNERSHIP_ATTR,
        placement_pixel_center_ownership_attr=(
            CROP_PLACEMENT_PADDED_PIXEL_CENTER_OWNERSHIP_ATTR
        ),
        placement_pixel_edge_ownership_attr=(
            CROP_PLACEMENT_PADDED_PIXEL_EDGE_OWNERSHIP_ATTR
        ),
        _root=root,
        _rowset_node=crop,
        _placement_node=crop["source_crop_xywh"],
        _roi_images_node=None,
        _verification_seal=_BOUND_SEALED_GEOMETRY_CROP_SOURCE_SEAL,
    )


def load_sealed_geometry_crop_source(
    root: Any,
    crop_path: str,
) -> BoundSealedGeometryCropSource:
    """Resolve the sealed geometry-only profile without successor-specific state.

    This is the geometry-only branch of the shared keypoint/subject-mask crop
    source resolver.  It validates the immutable publication, exact topology,
    lifecycle polarity, crop policy, live acquisition authority, camera frames,
    and ROI geometry.  It does not synthesize pixels and therefore cannot pass
    the direct-pixel inference gate.
    """

    path = str(crop_path or "").strip("/")
    parts = path.split("/")
    if len(parts) != 2 or parts[0] != "crop_runs" or not parts[1]:
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop resolution requires one crop_runs/<run> path."
        )
    identity = archive_identity(root)
    if identity.kind != "local_store_root" or not identity.key:
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop resolution requires one stable local archive root."
        )
    publication = open_persisted_crop_geometry_publication(
        Path(str(identity.key[0])),
        run_id=parts[1],
    )
    manifest = dict(publication.manifest)
    errors = validate_crop_run_manifest(manifest)
    if errors:
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop manifest is invalid: " + "; ".join(errors)
        )
    if manifest.get("schema_version") != CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION:
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop resolution requires coordinate manifest v2."
        )
    try:
        crop = root[path]
    except (KeyError, TypeError) as exc:
        raise SealedGeometryCropProfileError(
            f"Sealed geometry crop rowset cannot be opened: {exc}"
        ) from exc
    if canonical_node_path(crop) != path:
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop resolved to an unexpected archive path."
        )
    attrs = getattr(crop, "attrs", {})
    legacy_lineage_attrs = (
        "detection_acquisition_frame_mapping",
        "crop_geometry_selection",
        "collection_proxy_coordinate_successor_mapping",
    )
    payload = _require_mapping(manifest["payload"], label="manifest payload")
    publication_record = _require_mapping(
        payload["publication"], label="publication record"
    )
    if (
        attrs.get(CROP_RUN_MANIFEST_ATTRIBUTE) != manifest
        or any(name in attrs for name in legacy_lineage_attrs)
        or attrs.get("status") != "complete"
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("artifact_class") != "geometry_only_analysis"
        or attrs.get("production_candidate") is not True
        or attrs.get("immutable_snapshot") is not True
        or attrs.get("coordinate_contract") is not None
        or attrs.get("crop_storage_mode") not in (None, "geometry_only")
        or publication_record.get("completion_status") != "complete"
        or publication_record.get("stage_selector_eligible") is not False
        or publication_record.get("artifact_class") != "geometry_only_analysis"
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop lifecycle, profile identity, or lineage exclusivity is invalid."
        )
    parent_selection = root["crop_runs"].attrs.get("selection_contract")
    if parent_selection not in (None, "none_shadow_direct_path_only"):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop parent has an unsupported selection contract."
        )
    expected_names = set(CROP_GEOMETRY_SCHEMA_V1.binding_paths)
    if set(crop.keys()) != expected_names or set(publication.arrays) != expected_names:
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop does not expose the exact 13-array profile topology."
        )

    logical = _require_mapping(payload["logical_schema"], label="logical schema")
    dimensions = _require_mapping(logical["dimensions"], label="dimensions")
    n_instances = int(dimensions["n_instances"])
    n_frames = int(dimensions["n_frames"])
    source_width = int(dimensions["source_width"])
    source_height = int(dimensions["source_height"])
    try:
        crop_policy = crop_geometry_policy_from_manifest(logical["crop_policy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop lacks a valid crop policy."
        ) from exc
    if (
        crop_policy.padding_mode is not CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME
        or crop_policy.size_mode is not CropSizeMode.FIXED_PER_RUN
        or crop_policy.placement_mode is not CropPlacementMode.VERIFIED_EXPLICIT_PER_ROW
        or not isinstance(crop_policy.placement_authority, Mapping)
        or payload.get("coordinate_contract") is None
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop policy is not the supported fixed, zero-padded, explicit-origin profile."
        )

    pixel = _require_mapping(payload["source_pixel_authority"], label="pixel authority")
    if (
        int(pixel["n_frames"]) != n_frames
        or int(pixel["source_width"]) != source_width
        or int(pixel["source_height"]) != source_height
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop pixel dimensions differ from its logical schema."
        )
    _, acquisition = load_persisted_acquisition_camera_authority(root)
    camera_id = str(pixel["camera_identity"])
    if (
        acquisition.record.camera_id != camera_id
        or int(acquisition.record.source_total_frames) != n_frames
        or (
            pixel.get("recording_identity") is not None
            and getattr(acquisition.record, "recording_id", None) is not None
            and pixel["recording_identity"] != acquisition.record.recording_id
        )
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop pixel authority disagrees with live acquisition authority."
        )
    point_camera = load_source_camera_pixel_frame_authority(
        root[
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_POINT_PIXEL_CONVENTION}"
        ],
        acquisition_frame=acquisition,
    )
    bbox_camera = load_source_camera_pixel_frame_authority(
        root[
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_BBOX_PIXEL_CONVENTION}"
        ],
        acquisition_frame=acquisition,
    )
    if any(
        (
            frame.endpoint.width != source_width
            or frame.endpoint.height != source_height
        )
        for frame in (point_camera, bbox_camera)
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop source-camera extent differs from its live frames."
        )

    crop_arrays = publication.arrays
    roi_sizes = _array(crop_arrays["roi_sizes_full"], label="roi_sizes_full")
    if roi_sizes.dtype != np.dtype("<i4") or roi_sizes.shape != (n_instances, 2):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop ROI sizes are not exact int32 row-aligned values."
        )
    fixed_size = crop_policy.fixed_size_wh
    if fixed_size is None or any(int(value) <= 0 for value in fixed_size):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop policy lacks one positive fixed ROI extent."
        )
    roi_width, roi_height = (int(fixed_size[0]), int(fixed_size[1]))
    if n_instances and (
        np.any(roi_sizes <= 0)
        or not np.all(roi_sizes == np.asarray([roi_width, roi_height], dtype="<i4"))
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop ROI sizes differ from its fixed policy extent."
        )
    placement = _array(crop_arrays["source_crop_xywh"], label="source_crop_xywh")
    origins = _array(crop_arrays["roi_coordinates_full"], label="roi_coordinates_full")
    if (
        placement.dtype != np.dtype("<f4")
        or placement.shape != (n_instances, 4)
        or origins.dtype != np.dtype("<i4")
        or origins.shape != (n_instances, 2)
        or not np.array_equal(origins, np.rint(placement[:, :2]).astype("<i4", copy=False))
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop placement and explicit-origin arrays disagree."
        )
    return _build_bound_source(
        root=root,
        crop=crop,
        crop_path=path,
        manifest=manifest,
        n_instances=n_instances,
        roi_width=roi_width,
        roi_height=roi_height,
        point_camera=point_camera,
        bbox_camera=bbox_camera,
        acquisition_frame=acquisition,
    )


def require_bound_sealed_geometry_crop_source(
    value: Any,
    *,
    root: Any,
    crop_path: str,
) -> BoundSealedGeometryCropSource:
    """Revalidate a resolver-issued geometry-only crop source."""

    if (
        type(value) is not BoundSealedGeometryCropSource
        or value._seal is not _BOUND_SEALED_GEOMETRY_CROP_SOURCE_SEAL
    ):
        raise SealedGeometryCropProfileError(
            "A resolver-issued sealed geometry crop source is required."
        )
    path = str(crop_path or "").strip("/")
    if value.crop_path != path or archive_identity(value._root) != archive_identity(root):
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop source belongs to another path or archive."
        )
    current = load_sealed_geometry_crop_source(root, path)
    expected = (
        current.crop_geometry.selection_derivation.record_sha256,
        current.roi_geometry.derivation.record_sha256,
        current.crop_geometry.row_identity.leading_dimension,
        current.roi_frame.record_sha256,
        current.bbox_roi_frame.record_sha256,
        current.placement_ownership_attr,
        current.placement_pixel_center_ownership_attr,
        current.placement_pixel_edge_ownership_attr,
    )
    observed = (
        value.crop_geometry.selection_derivation.record_sha256,
        value.roi_geometry.derivation.record_sha256,
        value.crop_geometry.row_identity.leading_dimension,
        value.roi_frame.record_sha256,
        value.bbox_roi_frame.record_sha256,
        value.placement_ownership_attr,
        value.placement_pixel_center_ownership_attr,
        value.placement_pixel_edge_ownership_attr,
    )
    if observed != expected:
        raise SealedGeometryCropProfileError(
            "Sealed geometry crop source changed after it was resolved."
        )
    return value


def bind_sealed_geometry_crop_successor_source(
    *,
    analysis_zarr: Path,
    root: Any,
    crop_reference: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
    source_run_path: str,
    model_input_transform: ModelInputTransform,
) -> SealedGeometryCropSuccessorBinding:
    """Prove and bind the supported sealed geometry-only crop-v2 topology."""

    archive = analysis_zarr.expanduser().resolve()
    crop_path = str(crop_reference.get("run_path") or "").strip("/")
    if not crop_path.startswith("crop_runs/") or len(crop_path.split("/")) != 2:
        raise SealedGeometryCropProfileError(
            "Geometry-only crop bridge requires one exact crop_runs/<run> path."
        )
    run_id = crop_path.split("/", 1)[1]
    publication = open_persisted_crop_geometry_publication(archive, run_id=run_id)
    manifest = dict(publication.manifest)
    errors = validate_crop_run_manifest(manifest)
    if errors:
        raise SealedGeometryCropProfileError(
            "Geometry-only crop manifest is invalid: " + "; ".join(errors)
        )
    if manifest.get("schema_version") != CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop profile requires manifest v2 with its coordinate catalog."
        )
    payload = _require_mapping(manifest["payload"], label="manifest payload")
    logical = _require_mapping(payload["logical_schema"], label="logical schema")
    dimensions = _require_mapping(logical["dimensions"], label="dimensions")
    try:
        crop_policy = crop_geometry_policy_from_manifest(logical["crop_policy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop lacks a valid canonical crop_geometry_policy manifest."
        ) from exc
    if crop_policy.padding_mode is not CropPaddingMode.ZERO_OUTSIDE_SOURCE_FRAME:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only successor requires the producer "
            "zero_outside_source_frame crop policy."
        )
    if crop_policy.size_mode is not CropSizeMode.FIXED_PER_RUN:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only successor requires a fixed per-run crop extent."
        )
    if crop_policy.placement_mode is not CropPlacementMode.VERIFIED_EXPLICIT_PER_ROW:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only successor requires verified explicit per-row crop origins."
        )
    if not isinstance(crop_policy.placement_authority, Mapping):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop policy lacks its normalized explicit origin authority."
        )
    if payload.get("coordinate_contract") is None:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop lacks its coordinate catalog."
        )
    crop = root[crop_path]
    attrs = getattr(crop, "attrs", {})
    if (
        attrs.get("status") != "complete"
        or attrs.get("stage_selector_eligible") is not False
        or attrs.get("artifact_class") != "geometry_only_analysis"
        or attrs.get("coordinate_contract") is not None
        or attrs.get("crop_storage_mode") not in (None, "geometry_only")
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop does not have the exact complete geometry-only lifecycle."
        )
    child_names = set(crop.keys())
    expected_names = set(CROP_GEOMETRY_SCHEMA_V1.binding_paths)
    if child_names != expected_names or "roi_images" in child_names:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop arrays do not match the complete geometry-only topology."
        )
    _require_crop_reference(crop_reference, manifest=manifest, crop_path=crop_path)
    n_frames = int(dimensions["n_frames"])
    n_instances = int(dimensions["n_instances"])
    source_width = int(dimensions["source_width"])
    source_height = int(dimensions["source_height"])
    source_n_frames, source_n_instances, _ = _source_dimensions(source_manifest)
    if (source_n_frames, source_n_instances) != (n_frames, n_instances):
        raise SealedGeometryCropProfileError(
            "Source output dimensions do not match complete crop row coverage."
        )
    crop_arrays = publication.arrays
    expected_array_names = set(CROP_GEOMETRY_SCHEMA_V1.binding_paths)
    if set(crop_arrays) != expected_array_names:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop publication does not expose the complete exact array set."
        )
    _validate_source_selection(
        source_manifest=source_manifest,
        source_arrays=source_arrays,
        crop_arrays=crop_arrays,
        n_frames=n_frames,
        n_instances=n_instances,
        source_run_path=source_run_path,
    )
    pixel = _require_mapping(payload["source_pixel_authority"], label="pixel authority")
    if (
        int(pixel["n_frames"]) != n_frames
        or int(pixel["source_width"]) != source_width
        or int(pixel["source_height"]) != source_height
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop pixel authority dimensions differ from its manifest."
        )
    _, acquisition = load_persisted_acquisition_camera_authority(root)
    camera_id = str(pixel["camera_identity"])
    acquisition_recording_id = getattr(acquisition.record, "recording_id", None)
    if (
        pixel.get("recording_identity") is not None
        and acquisition_recording_id is not None
        and pixel.get("recording_identity") != acquisition_recording_id
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop pixel authority recording differs from acquisition authority."
        )
    if acquisition.record.camera_id != camera_id:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop pixel authority camera differs from acquisition authority."
        )
    point_camera = load_source_camera_pixel_frame_authority(
        root[
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_POINT_PIXEL_CONVENTION}"
        ],
        acquisition_frame=acquisition,
    )
    bbox_camera = load_source_camera_pixel_frame_authority(
        root[
            "analysis/coordinate_frames/source_camera/"
            f"{camera_id}/{SOURCE_CAMERA_BBOX_PIXEL_CONVENTION}"
        ],
        acquisition_frame=acquisition,
    )
    if (
        point_camera.pixel_convention != SOURCE_CAMERA_POINT_PIXEL_CONVENTION
        or bbox_camera.pixel_convention != SOURCE_CAMERA_BBOX_PIXEL_CONVENTION
        or point_camera.endpoint.width != source_width
        or point_camera.endpoint.height != source_height
        or bbox_camera.endpoint.width != source_width
        or bbox_camera.endpoint.height != source_height
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop source-camera extent or point/bbox convention is incompatible."
        )
    roi_sizes = _array(crop_arrays["roi_sizes_full"], label="roi_sizes_full")
    if roi_sizes.dtype != np.dtype("<i4") or roi_sizes.shape != (n_instances, 2):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop ROI sizes are not the exact int32 row-aligned array."
        )
    if np.any(roi_sizes <= 0) or not np.all(roi_sizes == np.asarray([roi_sizes[0]])):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop ROI sizes are not one exact constant positive extent."
        )
    roi_width, roi_height = (int(roi_sizes[0, 0]), int(roi_sizes[0, 1]))
    if crop_policy.fixed_size_wh != (roi_width, roi_height):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop policy fixed_size_wh differs from the actual ROI extent."
        )
    placement = _array(crop_arrays["source_crop_xywh"], label="source_crop_xywh")
    origins = _array(crop_arrays["roi_coordinates_full"], label="roi_coordinates_full")
    if origins.dtype != np.dtype("<i4") or origins.shape != (n_instances, 2):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop ROI origins are not the exact little-endian int32 row-aligned array."
        )
    if not np.array_equal(
        origins,
        np.rint(placement[:, :2]).astype("<i4", copy=False),
    ):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop ROI origins do not exactly equal source_crop_xywh requested origins."
        )
    padded_lineage = _padded_lineage_summary(
        placement,
        source_width=source_width,
        source_height=source_height,
        roi_width=roi_width,
        roi_height=roi_height,
        crop_policy_payload_digest=crop_policy.payload_digest,
        origin_authority_digest=canonical_json_sha256(crop_policy.placement_authority),
        provider_record_sha256=str(
            crop_policy.placement_authority["provider_record_sha256"]
        ),
    )
    if type(model_input_transform) is not ModelInputTransform:
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop successor requires one exact model input transform."
        )
    if model_input_transform.native_shape != (roi_height, roi_width):
        raise SealedGeometryCropProfileError(
            "Sealed geometry-only crop ROI extent differs from the exact model input transform."
        )
    source = _build_bound_source(
        root=root,
        crop=crop,
        crop_path=crop_path,
        manifest=manifest,
        n_instances=n_instances,
        roi_width=roi_width,
        roi_height=roi_height,
        point_camera=point_camera,
        bbox_camera=bbox_camera,
        acquisition_frame=acquisition,
    )
    logical_document = _require_mapping(
        payload["logical_content"], label="logical content"
    )
    declarations = _require_mapping(
        logical_document["document"], label="logical content document"
    )["arrays"]
    row_digest = _require_mapping(declarations, label="logical content arrays")[
        "source_row_signature"
    ]["sha256"]
    coordinate_catalog_digest = _require_mapping(
        payload["coordinate_contract"], label="coordinate contract"
    )["digest"]
    return SealedGeometryCropSuccessorBinding(
        source=source,
        crop_path=crop_path,
        manifest=manifest,
        manifest_digest=canonical_json_sha256(manifest),
        manifest_payload_digest=str(manifest["payload_digest"]),
        logical_content_digest=str(logical_document["digest"]),
        row_signatures_digest=str(row_digest),
        coordinate_catalog_digest=str(coordinate_catalog_digest),
        n_frames=n_frames,
        n_instances=n_instances,
        source_width=source_width,
        source_height=source_height,
        source_run_path=source_run_path,
        padded_lineage=padded_lineage,
        successor_evidence_record={
            "schema_id": SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_SCHEMA_ID,
            "schema_version": SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_SCHEMA_VERSION,
            "adapter_kind": SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_KIND,
            "crop_path": crop_path,
            "crop_manifest_digest": canonical_json_sha256(manifest),
            "crop_manifest_payload_digest": manifest["payload_digest"],
            "crop_logical_content_digest": logical_document["digest"],
            "crop_row_signatures_digest": row_digest,
            "crop_coordinate_catalog_digest": coordinate_catalog_digest,
            "source_run_path": source_run_path,
            "source_camera_identity": camera_id,
            "source_camera_extent": {"width": source_width, "height": source_height},
            "point_convention": SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
            "bbox_convention": SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
            "roi_extent": {"width": roi_width, "height": roi_height},
            "model_input_transform": model_input_transform.to_attrs(),
            "pixel_source": "none_historical_coordinate_migration_only",
            "ephemeral_flat_pixel_cache": "not_an_immutable_source",
            "validated_array_digest_basis": "crop_manifest_logical_content_v1",
            "validated_source_rowset": True,
            "padded_crop_lineage": padded_lineage,
            "crop_policy_payload_digest": crop_policy.payload_digest,
            "origin_authority_digest": canonical_json_sha256(
                crop_policy.placement_authority
            ),
            "origin_authority": copy.deepcopy(dict(crop_policy.placement_authority)),
        },
    )

__all__ = [
    "BoundSealedGeometryCropSource",
    "SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_KIND",
    "SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_SCHEMA_ID",
    "SEALED_GEOMETRY_CROP_SUCCESSOR_EVIDENCE_SCHEMA_VERSION",
    "SealedGeometryCropProfileError",
    "SealedGeometryCropSuccessorBinding",
    "bind_persisted_padded_placement_record",
    "bind_persisted_run_attribute_record",
    "bind_sealed_geometry_bbox_normalization_to_successor",
    "bind_sealed_geometry_crop_successor_source",
    "load_sealed_geometry_crop_source",
    "load_sealed_geometry_bbox_normalization_from_successor",
    "stamp_persisted_padded_placement_provenance",
]
