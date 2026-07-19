"""Immutable crop-row identity for keyed downstream compaction.

Modern materialized crops persist ``source_row_signature``.  Historical
clipped-collection proxy crops intentionally use the auxiliary completion
state and predate that array.  This module keeps the compatibility decision
explicit: only a fully declared, geometry-only proxy may be treated as an
immutable crop snapshot, and its signatures use a versioned bootstrap spec
whose trust boundary is the finalized clip collection plus canonical crop
geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fisheye.shared.row_source_signature import (
    ROW_SOURCE_SIGNATURE_ARRAY,
    RowSourceSignatureSpec,
    build_row_source_signatures,
    load_row_source_signature_spec,
    validate_row_source_signature_array,
)
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETION_STATUS_ATTR,
    is_run_complete_in_parent,
)


LEGACY_PROXY_SIGNATURE_SCHEMA_ID = "palette.legacy_proxy_crop_signature_bootstrap"
LEGACY_PROXY_SIGNATURE_SCHEMA_VERSION = 1
LEGACY_PROXY_SIGNATURE_STAGE = "crop"
LEGACY_PROXY_COMPLETION_STATUS = "auxiliary"
LEGACY_PROXY_SCHEMAS = frozenset(
    {
        "palette_clipped_collection_proxy_crop_run_v1",
        "palette_clipped_collection_merged_proxy_crop_run_v1",
    }
)
_BOOTSTRAP_COMPONENTS = (
    ("frame_indices", np.int64, ()),
    ("roi_coordinates_full", np.int32, (2,)),
    ("source_clip_indices", np.int64, ()),
    ("source_clip_local_frame_indices", np.int64, ()),
)


class CropSnapshotIdentityError(RuntimeError):
    """Raised when a crop cannot be used as an immutable keyed snapshot."""


@dataclass(frozen=True)
class CropSignatureSnapshot:
    """Resolved row signatures and the contract that produced them."""

    signatures: np.ndarray
    spec: RowSourceSignatureSpec
    mode: str


def _text(value: object) -> str:
    return str(value or "").strip()


def is_verified_auxiliary_proxy_crop(group: Any) -> bool:
    """Return whether ``group`` satisfies the immutable proxy-crop contract."""

    attrs = group.attrs
    return bool(
        _text(attrs.get(RUN_COMPLETION_STATUS_ATTR)).lower()
        == LEGACY_PROXY_COMPLETION_STATUS
        and attrs.get("proxy_crop_complete") is True
        and attrs.get("stage_selector_eligible") is False
        and _text(attrs.get("crop_storage_mode")) == "geometry_only"
        and _text(attrs.get("stage")) == "crop_proxy"
        and _text(attrs.get("schema")) in LEGACY_PROXY_SCHEMAS
        and _text(attrs.get("source_collection_id"))
        and _text(attrs.get("source_collection_path"))
    )


def require_crop_snapshot(parent: Any, name: str, *, label: str) -> Any:
    """Resolve a complete crop or verified auxiliary collection proxy."""

    if name not in parent:
        raise CropSnapshotIdentityError(f"{label} {name!r} does not exist.")
    group = parent[name]
    if not (
        is_run_complete_in_parent(parent, group)
        or is_verified_auxiliary_proxy_crop(group)
    ):
        raise CropSnapshotIdentityError(
            f"{label} {name!r} is neither complete nor a verified auxiliary proxy."
        )
    return group


def _positive_int(value: object, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CropSnapshotIdentityError(f"Crop attr {name!r} must be an integer.") from exc
    if result <= 0:
        raise CropSnapshotIdentityError(f"Crop attr {name!r} must be positive.")
    return result


def _shape_attr(group: Any, *names: str) -> tuple[int, int]:
    for name in names:
        raw = group.attrs.get(name)
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return (
                _positive_int(raw[0], name=f"{name}[0]"),
                _positive_int(raw[1], name=f"{name}[1]"),
            )
    raise CropSnapshotIdentityError(
        f"Crop lacks a valid two-value shape attr among {names!r}."
    )


def _recording_identity(root: Any) -> str:
    for name in ("recording_id", "session_id", "recording_name"):
        value = _text(root.attrs.get(name))
        if value:
            return value
    raise CropSnapshotIdentityError(
        "Legacy proxy signature bootstrap requires canonical recording identity."
    )


def _bootstrap_context(root: Any, crop: Any) -> dict[str, object]:
    frame_shape = (
        _positive_int(
            crop.attrs.get("height", root.attrs.get("height")),
            name="height",
        ),
        _positive_int(
            crop.attrs.get("width", root.attrs.get("width")),
            name="width",
        ),
    )
    return {
        "bootstrap_schema_id": LEGACY_PROXY_SIGNATURE_SCHEMA_ID,
        "bootstrap_schema_version": LEGACY_PROXY_SIGNATURE_SCHEMA_VERSION,
        "trust_boundary": "finalized_clip_collection_and_proxy_geometry",
        "recording_identity": _recording_identity(root),
        "source_collection_id": _text(crop.attrs.get("source_collection_id")),
        "source_collection_path": _text(crop.attrs.get("source_collection_path")),
        "source_frame_shape": list(frame_shape),
        "roi_shape": list(_shape_attr(crop, "roi_shape", "roi_size")),
        "crop_policy": _text(crop.attrs.get("crop_policy")),
        "bbox_norm_coords_semantics": _text(
            crop.attrs.get("bbox_norm_coords_semantics")
        ),
        "padding": "zero_outside_source_frame_bounds",
        "output_representation": "uint8_grayscale_roi",
    }


def _bootstrap_proxy_signatures(
    root: Any,
    crop: Any,
    *,
    label: str,
    instance_keys: np.ndarray | None = None,
) -> CropSignatureSnapshot:
    if not is_verified_auxiliary_proxy_crop(crop):
        raise CropSnapshotIdentityError(
            f"{label} lacks persisted source signatures and is not an eligible legacy proxy."
        )
    if instance_keys is None:
        if "instance_key" not in crop:
            raise CropSnapshotIdentityError(
                f"{label} lacks instance_key and no exact manifest keys were supplied."
            )
        keys = np.asarray(crop["instance_key"][:], dtype=np.uint64).reshape(-1)
    else:
        keys = np.asarray(instance_keys, dtype=np.uint64).reshape(-1)
    if np.unique(keys).shape[0] != keys.shape[0]:
        raise CropSnapshotIdentityError(f"{label}/instance_key is not unique.")
    content: dict[str, np.ndarray] = {}
    for name, dtype, trailing_shape in _BOOTSTRAP_COMPONENTS:
        if name not in crop:
            raise CropSnapshotIdentityError(
                f"{label} lacks legacy bootstrap component {name!r}."
            )
        values = np.asarray(crop[name][:], dtype=dtype)
        expected = (keys.shape[0], *trailing_shape)
        if tuple(values.shape) != expected:
            raise CropSnapshotIdentityError(
                f"{label}/{name} must have shape {expected}, got {values.shape}."
            )
        content[name] = values
    batch = build_row_source_signatures(
        stage=LEGACY_PROXY_SIGNATURE_STAGE,
        instance_keys=keys,
        content_components=content,
        compatibility_context=_bootstrap_context(root, crop),
    )
    return CropSignatureSnapshot(
        signatures=batch.signatures,
        spec=batch.spec,
        mode="legacy_proxy_bootstrap_v1",
    )


def resolve_crop_source_signatures(
    root: Any,
    crop: Any,
    *,
    label: str,
    instance_keys: np.ndarray | None = None,
) -> CropSignatureSnapshot:
    """Load persisted signatures or derive the explicit legacy proxy baseline."""

    if ROW_SOURCE_SIGNATURE_ARRAY not in crop:
        return _bootstrap_proxy_signatures(
            root,
            crop,
            label=label,
            instance_keys=instance_keys,
        )
    if "instance_key" in crop:
        rows = int(crop["instance_key"].shape[0])
    elif "frame_indices" in crop:
        rows = int(crop["frame_indices"].shape[0])
    else:
        rows = -1
    validate_row_source_signature_array(
        crop[ROW_SOURCE_SIGNATURE_ARRAY], expected_row_count=rows
    )
    return CropSignatureSnapshot(
        signatures=np.asarray(crop[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8),
        spec=load_row_source_signature_spec(crop.attrs),
        mode="persisted",
    )


def crop_frame_counts(
    root: Any,
    crop: Any,
    *,
    fallback_frame_count: int | None = None,
) -> np.ndarray:
    """Return canonical frame counts, deriving proxy counts when absent."""

    if "frame_counts" in crop:
        return np.asarray(crop["frame_counts"][:], dtype=np.int32).reshape(-1)
    if "frame_indices" not in crop:
        raise CropSnapshotIdentityError("Crop lacks frame_indices.")
    frames = np.asarray(crop["frame_indices"][:], dtype=np.int64).reshape(-1)
    candidates: list[int] = []
    if fallback_frame_count is not None:
        candidates.append(int(fallback_frame_count))
    for value in (
        root.attrs.get("total_frames"),
        (root.attrs.get("source_video_metadata") or {}).get("total_frames")
        if isinstance(root.attrs.get("source_video_metadata"), dict)
        else None,
    ):
        try:
            if value is not None:
                candidates.append(int(value))
        except (TypeError, ValueError):
            continue
    minimum = int(frames.max()) + 1 if frames.size else 0
    total_frames = max([minimum, *candidates], default=minimum)
    if total_frames < 0:
        raise CropSnapshotIdentityError("Resolved frame count is negative.")
    return np.bincount(frames, minlength=total_frames).astype(np.int32, copy=False)


def crop_detection_source(crop: Any) -> np.ndarray:
    """Return row-aligned detection-source labels; proxy default is YOLO (0)."""

    if "instance_key" in crop:
        rows = int(crop["instance_key"].shape[0])
    elif "frame_indices" in crop:
        rows = int(crop["frame_indices"].shape[0])
    else:
        rows = -1
    if "detection_source" not in crop:
        if not is_verified_auxiliary_proxy_crop(crop):
            raise CropSnapshotIdentityError("Non-proxy crop lacks detection_source.")
        return np.zeros(rows, dtype=np.int8)
    values = np.asarray(crop["detection_source"][:], dtype=np.int8).reshape(-1)
    if values.shape != (rows,):
        raise CropSnapshotIdentityError("Crop detection_source is not row-aligned.")
    return values


__all__ = [
    "CropSignatureSnapshot",
    "CropSnapshotIdentityError",
    "LEGACY_PROXY_SIGNATURE_SCHEMA_ID",
    "LEGACY_PROXY_SIGNATURE_SCHEMA_VERSION",
    "crop_detection_source",
    "crop_frame_counts",
    "is_verified_auxiliary_proxy_crop",
    "require_crop_snapshot",
    "resolve_crop_source_signatures",
]
