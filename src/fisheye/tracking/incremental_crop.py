"""Keyed copy-forward materialization for complete immutable crop runs.

This is the Phase-1 reference adapter for incremental materialization.  It is
intentionally single-writer: compact identity/geometry and the keyed plan are
held in memory, while dense ROI pixels are copied or computed one complete
output chunk at a time.  A new run is selected only after exact source
revalidation and complete payload readback.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_NOT_PUBLISHED,
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    AcquisitionPublicationStatusError,
    MATERIALIZED_ACQUISITION_AUTHORITY_MODE,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.archive_identity import (
    ArchiveIdentityError,
    require_same_archive,
)
from fisheye.shared.composite_crop import (
    COMPOSITE_CROP_PAYLOAD_GROUP,
    COMPOSITE_CROP_SCHEMA_ID,
    COMPOSITE_CROP_SCHEMA_VERSION,
    COMPOSITE_CROP_SOURCE_BASE,
    COMPOSITE_CROP_SOURCE_DELTA,
    COMPOSITE_CROP_STORAGE_MODE,
    validate_composite_crop_run,
)
from fisheye.shared.coordinate_reference import (
    bind_array_reference_extent,
    canonical_node_path,
)
from fisheye.shared.directed_transform_chain import (
    resolve_bound_directed_transform_chain,
)
from fisheye.shared.directed_transform_v2 import (
    DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    stamp_directed_transform_v2,
)
from fisheye.shared.crop_roi_layout import (
    DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    build_canonical_crop_roi_layout,
    build_crop_roi_create_kwargs,
    crop_roi_layout_attrs,
)
from fisheye.shared.keyed_delta import (
    ACTION_CODE_MAP,
    KeyedDeltaPlan,
    build_keyed_delta_plan,
    write_keyed_delta_plan,
)
from fisheye.shared.observation_coordinate_publication import (
    BoundDetectionObservationGeometry,
    _load_persisted_ordinary_crop_observation_geometry,
    capture_observation_coordinate_publication_checkpoint,
    detection_observation_geometry_values,
    load_persisted_detection_observation_geometry,
    publish_crop_roi_bbox_edge_reference_extent,
    publish_crop_observation_geometry,
    publish_crop_roi_geometry,
    restore_observation_coordinate_publication_checkpoint,
)
from fisheye.shared.pixel_frame_authority import (
    CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    load_persisted_acquisition_camera_authority,
    stamp_crop_placement_ownership,
    stamp_roi_pixel_frame_authority,
)
from fisheye.shared.roi_pixel_contract import (
    CENTER_ROUNDING_NP_ROUND,
    ROI_IMAGE_REPRESENTATION,
    crop_run_pixel_contract,
)
from fisheye.shared.row_source_signature import (
    ROW_SOURCE_SIGNATURE_ARRAY,
    RowSourceSignatureSpec,
    build_row_source_signatures,
    load_row_source_signature_spec,
    validate_row_source_signature_array,
)
from fisheye.shared.rowset_fingerprint import (
    RowsetFingerprint,
    assert_rowset_fingerprint_matches,
    build_group_rowset_fingerprint,
)
from fisheye.shared.transform_authority import (
    TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    stamp_crop_placement_transform_authority,
)
from fisheye.shared.run_provenance import (
    RUN_PROVENANCE_ATTR,
    validate_run_provenance,
)
from fisheye.shared.zarr.columnar import store_array
from fisheye.shared.zarr_run_completion import (
    RUN_COMPLETED_AT_ATTR,
    RUN_COMPLETION_STATUS_ATTR,
    RUN_LATEST_COMPLETE_ATTR,
    RUN_LATEST_PENDING_ATTR,
    RUN_STATUS_COMPLETE,
    mark_run_complete,
    mark_run_failed,
    mark_run_started,
    note_pending_latest,
    require_runs_parent,
)


INCREMENTAL_CROP_SCHEMA_ID = "palette.incremental_crop_materialization"
INCREMENTAL_CROP_SCHEMA_VERSION = 1
INCREMENTAL_CROP_SIGNATURE_STAGE = "crop"
INCREMENTAL_CROP_WRITER_OWNERSHIP = "single_driver_complete_physical_chunks"
INCREMENTAL_CROP_PUBLICATION_POLICY = (
    "serialized_parent_writer_validate_then_single_parent_attrs_update_v1"
)
DEFAULT_SIGNATURE_BATCH_ROWS = 16_384
DEFAULT_TABULAR_SHARD_ROWS = 131_072
CROP_COORDINATE_CONTRACT_MODES = ("canonical",)
HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE = "historical_migration_composite"
HISTORICAL_COMPOSITE_COORDINATE_CONTRACT = (
    "historical_composite_unbound_v1"
)
_CROP_SELECTOR_ATTRS = (
    "latest",
    RUN_LATEST_COMPLETE_ATTR,
    RUN_LATEST_PENDING_ATTR,
    "latest_materialized",
    "latest_any",
    "latest_composite",
    "authoritative_run",
    "authoritative_run_provenance",
    "publication_generation",
    "publication_policy",
)
OPTIONAL_SOURCE_ROW_ARRAYS = (
    "source_frame_indices",
    "source_clip_indices",
    "source_clip_local_frame_indices",
    "source_refined_row_ids",
    "source_detect_row_index",
    "refined_row_id",
    "detection_source",
)


class IncrementalCropError(RuntimeError):
    """Raised when an incremental crop cannot be proven complete and safe."""


@dataclass(frozen=True)
class CropSourceSnapshot:
    """Compact source state captured at one committed authoring boundary."""

    source_path: str
    instance_keys: np.ndarray
    frame_indices: np.ndarray
    bbox_norm_coords: np.ndarray
    optional_row_arrays: Mapping[str, np.ndarray]
    signatures: np.ndarray
    signature_spec: RowSourceSignatureSpec
    rowset_fingerprint: RowsetFingerprint

    @property
    def row_count(self) -> int:
        return int(self.instance_keys.shape[0])


@dataclass(frozen=True)
class IncrementalCropResult:
    """Auditable result of one complete crop materialization."""

    run_name: str
    plan: KeyedDeltaPlan
    copied_rows: int
    computed_rows: int
    omitted_rows: int
    roi_payload_bytes_written: int
    roi_payload_bytes_read_from_base: int
    source_frame_bytes_read: int
    validation_readback_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "copied_rows": self.copied_rows,
            "computed_rows": self.computed_rows,
            "omitted_rows": self.omitted_rows,
            "roi_payload_bytes_written": self.roi_payload_bytes_written,
            "roi_payload_bytes_read_from_base": self.roi_payload_bytes_read_from_base,
            "source_frame_bytes_read": self.source_frame_bytes_read,
            "validation_readback_bytes": self.validation_readback_bytes,
            "plan": self.plan.summary(),
        }


@dataclass(frozen=True)
class ResolvedIncrementalCropCoordinateContext:
    """Exact producer/frame inputs selected for one crop coordinate regime."""

    coordinate_contract_mode: str
    source_group: Any
    frame_source: Any
    source_geometry: BoundDetectionObservationGeometry | None
    source_pixel_fingerprint: str
    authority_signature: tuple[str, ...] | None


def _acquisition_publication_status_is_absent(root: Any) -> bool:
    root_attrs = getattr(root, "attrs", {})
    raw = root.get("raw_video")
    raw_attrs = getattr(raw, "attrs", {})
    return (
        ACQUISITION_AUTHORITY_STATUS_ATTR not in root_attrs
        and ACQUISITION_AUTHORITY_STATUS_ATTR not in raw_attrs
    )


def _has_uncommitted_acquisition_authority_artifacts(root: Any) -> bool:
    analysis = root.get("analysis")
    authorities = (
        analysis.get("acquisition_camera_frames")
        if analysis is not None
        else None
    )
    if authorities is not None and bool(tuple(authorities.keys())):
        return True
    raw = root.get("raw_video")
    manifests = raw.get("manifests") if raw is not None else None
    return manifests is not None and "images_full_materialization" in manifests


def _require_explicit_legacy_crop_regime(root: Any) -> None:
    if _acquisition_publication_status_is_absent(root):
        if _has_uncommitted_acquisition_authority_artifacts(root):
            raise IncrementalCropError(
                "Explicit legacy crop found status-less acquisition authority "
                "artifacts; this is incomplete publication, not a historical "
                "noncanonical archive."
            )
        return
    try:
        status = load_acquisition_authority_publication_status(root)
    except AcquisitionPublicationStatusError as exc:
        raise IncrementalCropError(
            "Explicit legacy crop found malformed or conflicting acquisition "
            "publication state."
        ) from exc
    if status.status != ACQUISITION_AUTHORITY_NOT_PUBLISHED:
        raise IncrementalCropError(
            "Explicit legacy crop cannot bypass a pending or published "
            f"acquisition authority (status={status.status!r})."
        )
    if _has_uncommitted_acquisition_authority_artifacts(root):
        raise IncrementalCropError(
            "Explicit noncanonical acquisition status conflicts with persisted "
            "acquisition authority artifacts; explicit repair is required."
        )


def _exact_root_node(root: Any, supplied: Any, path: str, *, label: str) -> Any:
    try:
        resolved = root[path]
    except Exception as exc:
        raise IncrementalCropError(
            f"Canonical crop requires exact persisted {label} at {path!r}."
        ) from exc
    if canonical_node_path(resolved) != path or canonical_node_path(supplied) != path:
        raise IncrementalCropError(
            f"Canonical crop {label} is not the exact root-owned node {path!r}."
        )
    try:
        require_same_archive(root, resolved, supplied)
    except ArchiveIdentityError as exc:
        raise IncrementalCropError(
            f"Canonical crop {label} comes from a different archive/store."
        ) from exc
    return resolved


def _materialization_id(ownership: Any) -> str:
    operation = ownership.record.import_operation
    manifest = (
        operation.get("materialization_manifest")
        if isinstance(operation, Mapping)
        else None
    )
    value = manifest.get("materialization_id") if isinstance(manifest, Mapping) else None
    if not isinstance(value, str) or len(value) != 64:
        raise IncrementalCropError(
            "Canonical crop acquisition ownership lacks an exact materialization id."
        )
    return value


def resolve_incremental_crop_coordinate_context(
    root: Any,
    *,
    source_group: Any,
    source_path: str,
    frame_source: Any,
    source_pixel_fingerprint: str,
    coordinate_contract_mode: str,
) -> ResolvedIncrementalCropCoordinateContext:
    """Select one exact complete canonical detection and acquisition authority."""

    if coordinate_contract_mode not in CROP_COORDINATE_CONTRACT_MODES:
        raise IncrementalCropError(
            "Unsupported coordinate_contract_mode "
            f"{coordinate_contract_mode!r}; expected one of "
            f"{CROP_COORDINATE_CONTRACT_MODES}."
        )
    source_label = str(source_path).strip().strip("/")
    if not source_label:
        raise IncrementalCropError("source_path must be non-empty.")
    fingerprint = str(source_pixel_fingerprint).strip()
    if not fingerprint:
        raise IncrementalCropError("source_pixel_fingerprint must be non-empty.")

    try:
        status = load_acquisition_authority_publication_status(root)
    except AcquisitionPublicationStatusError as exc:
        raise IncrementalCropError(
            "Canonical crop requires exact mirrored acquisition publication state."
        ) from exc
    if status.status != ACQUISITION_AUTHORITY_PUBLISHED:
        raise IncrementalCropError(
            "Canonical crop requires a published acquisition authority; "
            f"found status={status.status!r}, reason={status.reason_code!r}."
        )
    if status.authority_mode != MATERIALIZED_ACQUISITION_AUTHORITY_MODE:
        raise IncrementalCropError(
            "Canonical materialized crop requires materialized_source_frames_v1 "
            f"acquisition authority, not {status.authority_mode!r}."
        )
    ownership, acquisition = load_persisted_acquisition_camera_authority(root)
    expected_authority_path = (
        f"analysis/acquisition_camera_frames/{acquisition.record.camera_id}"
    )
    if (
        ownership.record.mode != status.authority_mode
        or status.authority_path != expected_authority_path
    ):
        raise IncrementalCropError(
            "Published acquisition status mode/path disagrees with the exact "
            "persisted materialization authority."
        )
    ownership.assert_verified()
    acquisition.assert_verified()
    exact_materialization_id = _materialization_id(ownership)
    if fingerprint != exact_materialization_id:
        raise IncrementalCropError(
            "Canonical crop source_pixel_fingerprint must equal the exact "
            "acquisition materialization id."
        )
    exact_frame_source = _exact_root_node(
        root,
        frame_source,
        "raw_video/images_full",
        label="frame_source",
    )
    source_parts = source_label.split("/")
    if len(source_parts) != 2 or source_parts[0] != "detect_runs":
        raise IncrementalCropError(
            "Canonical crop sources must be exact detect_runs/<run> rowsets."
        )
    try:
        detection_parent = root["detect_runs"]
    except Exception as exc:
        raise IncrementalCropError(
            "Canonical crop requires the root-owned detect_runs parent."
        ) from exc
    selected_name = detection_parent.attrs.get("authoritative_run")
    selected_attr = "authoritative_run"
    if not isinstance(selected_name, str) or not selected_name.strip():
        selected_name = detection_parent.attrs.get(RUN_LATEST_COMPLETE_ATTR)
        selected_attr = RUN_LATEST_COMPLETE_ATTR
    if not isinstance(selected_name, str) or not selected_name.strip():
        raise IncrementalCropError(
            "Canonical crop requires an explicit authoritative_run or "
            "latest_complete detection selector."
        )
    if source_parts[1] != selected_name.strip():
        raise IncrementalCropError(
            "Canonical crop source is not the exact selected detection: "
            f"{selected_attr}={selected_name!r}, source={source_label!r}."
        )
    exact_source_group = _exact_root_node(
        root,
        source_group,
        source_label,
        label="detection rowset",
    )
    source_geometry = load_persisted_detection_observation_geometry(
        root,
        source_label,
    )
    source_geometry.assert_verified()
    signature = (
        status.status,
        status.authority_mode or "",
        status.authority_path or "",
        ownership.record_ref,
        ownership.record_sha256,
        acquisition.record_ref,
        acquisition.record_sha256,
        exact_materialization_id,
        source_geometry.row_identity.record_ref,
        source_geometry.row_identity.record_sha256,
        source_geometry.bbox_projection.record_ref,
        source_geometry.bbox_projection.record_sha256,
        source_geometry.bbox_center_derivation.record_ref,
        source_geometry.bbox_center_derivation.record_sha256,
        source_geometry.bbox_normalized.descriptor.digest(),
        source_geometry.bbox_image.descriptor.digest(),
        source_geometry.centers_image.descriptor.digest(),
    )
    return ResolvedIncrementalCropCoordinateContext(
        coordinate_contract_mode=coordinate_contract_mode,
        source_group=exact_source_group,
        frame_source=exact_frame_source,
        source_geometry=source_geometry,
        source_pixel_fingerprint=exact_materialization_id,
        authority_signature=signature,
    )


def resolve_historical_composite_coordinate_context(
    root: Any,
    *,
    source_group: Any,
    source_path: str,
    frame_source: Any,
    source_pixel_fingerprint: str,
    coordinate_contract_mode: str,
) -> ResolvedIncrementalCropCoordinateContext:
    """Resolve an explicitly unbound historical composite migration input."""

    if coordinate_contract_mode != HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE:
        raise IncrementalCropError(
            "Historical composite crops require explicit "
            f"coordinate_contract_mode={HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE!r}."
        )
    source_label = str(source_path).strip().strip("/")
    fingerprint = str(source_pixel_fingerprint).strip()
    if not source_label or not fingerprint:
        raise IncrementalCropError(
            "Historical composite source path and pixel fingerprint are required."
        )
    _require_explicit_legacy_crop_regime(root)
    return ResolvedIncrementalCropCoordinateContext(
        coordinate_contract_mode=coordinate_contract_mode,
        source_group=source_group,
        frame_source=frame_source,
        source_geometry=None,
        source_pixel_fingerprint=fingerprint,
        authority_signature=None,
    )


def _normalize_roi_size(roi_size: tuple[int, int]) -> tuple[int, int]:
    if len(roi_size) != 2:
        raise IncrementalCropError("roi_size must contain (height, width).")
    normalized = (int(roi_size[0]), int(roi_size[1]))
    if min(normalized) <= 0:
        raise IncrementalCropError("roi_size dimensions must be positive.")
    return normalized


def _normalize_frame_shape(frame_shape: tuple[int, int]) -> tuple[int, int]:
    if len(frame_shape) != 2:
        raise IncrementalCropError(
            "Only two-dimensional grayscale source frames are supported."
        )
    normalized = (int(frame_shape[0]), int(frame_shape[1]))
    if min(normalized) <= 0:
        raise IncrementalCropError("Source-frame dimensions must be positive.")
    return normalized


def _require_source_arrays(
    source_group: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [
        name
        for name in ("instance_key", "frame_indices", "bbox_norm_coords")
        if name not in source_group
    ]
    if missing:
        raise IncrementalCropError(
            "Modern incremental crops require source arrays: " + ", ".join(missing)
        )
    raw_keys = np.asarray(source_group["instance_key"][:])
    if raw_keys.dtype.kind not in "iu":
        raise IncrementalCropError("Source instance_key must use an integer dtype.")
    if raw_keys.dtype.kind == "i" and np.any(raw_keys < 0):
        raise IncrementalCropError("Source instance_key values must be nonnegative.")
    keys = np.asarray(raw_keys, dtype=np.uint64).reshape(-1)
    if int(np.unique(keys).shape[0]) != int(keys.shape[0]):
        raise IncrementalCropError("Source instance_key values must be unique.")

    frames = np.asarray(source_group["frame_indices"][:])
    if frames.dtype.kind not in "iu":
        raise IncrementalCropError("Source frame_indices must use an integer dtype.")
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    boxes = np.asarray(source_group["bbox_norm_coords"][:])
    if boxes.ndim != 2 or int(boxes.shape[1]) != 4 or boxes.dtype.kind != "f":
        raise IncrementalCropError(
            "Source bbox_norm_coords must be a floating-point [N,4] array."
        )
    if frames.shape != keys.shape or int(boxes.shape[0]) != int(keys.shape[0]):
        raise IncrementalCropError(
            "Source identity, frame, and bbox row counts differ."
        )
    if np.any(frames < 0):
        raise IncrementalCropError("Source frame_indices contains negative values.")
    if not np.all(np.isfinite(boxes)):
        raise IncrementalCropError(
            "Source bbox_norm_coords contains non-finite values."
        )
    return keys, frames, boxes


def crop_signature_compatibility_context(
    *,
    source_pixel_fingerprint: str,
    source_rowset_family: str,
    frame_shape: tuple[int, int],
    roi_size: tuple[int, int],
) -> dict[str, Any]:
    """Return the exact global contract under which crop rows may be reused."""

    fingerprint = str(source_pixel_fingerprint).strip()
    if not fingerprint:
        raise IncrementalCropError("source_pixel_fingerprint must be non-empty.")
    source_family = str(source_rowset_family).strip().strip("/")
    if not source_family or "/" in source_family:
        raise IncrementalCropError(
            "source_rowset_family must be one non-empty group name."
        )
    height, width = _normalize_frame_shape(frame_shape)
    roi_height, roi_width = _normalize_roi_size(roi_size)
    return {
        "adapter_schema_id": INCREMENTAL_CROP_SCHEMA_ID,
        "adapter_schema_version": INCREMENTAL_CROP_SCHEMA_VERSION,
        "source_pixel_fingerprint": fingerprint,
        "source_rowset_family": source_family,
        "source_frame_shape": [height, width],
        "source_frame_dtype": "uint8",
        "source_frame_representation": "raw_video/images_full_grayscale",
        "roi_size": [roi_height, roi_width],
        "bbox_coordinate_contract": "normalized_cx_cy_width_height_xy_order",
        "center_rounding": CENTER_ROUNDING_NP_ROUND,
        "padding": "zero_outside_source_frame_bounds",
        "output_representation": ROI_IMAGE_REPRESENTATION,
    }


def capture_crop_source_snapshot(
    source_group: Any,
    *,
    source_path: str,
    source_pixel_fingerprint: str,
    frame_shape: tuple[int, int],
    roi_size: tuple[int, int],
    signature_batch_rows: int = DEFAULT_SIGNATURE_BATCH_ROWS,
) -> CropSourceSnapshot:
    """Capture compact source identity and bounded-batch crop signatures."""

    source_label = str(source_path).strip().strip("/")
    if not source_label:
        raise IncrementalCropError("source_path must be non-empty.")
    keys, frame_indices, boxes = _require_source_arrays(source_group)
    optional_row_arrays: dict[str, np.ndarray] = {}
    for name in OPTIONAL_SOURCE_ROW_ARRAYS:
        if name not in source_group:
            continue
        values = np.asarray(source_group[name][:])
        if values.ndim < 1 or int(values.shape[0]) != int(keys.shape[0]):
            raise IncrementalCropError(
                f"Optional source lineage array {name!r} is not row-aligned."
            )
        optional_row_arrays[name] = values
    batch_rows = int(signature_batch_rows)
    if batch_rows <= 0:
        raise IncrementalCropError("signature_batch_rows must be positive.")
    context = crop_signature_compatibility_context(
        source_pixel_fingerprint=source_pixel_fingerprint,
        source_rowset_family=source_label.split("/", maxsplit=1)[0],
        frame_shape=frame_shape,
        roi_size=roi_size,
    )
    signatures = np.empty((keys.shape[0], 32), dtype=np.uint8)
    expected_spec: RowSourceSignatureSpec | None = None
    slices = (
        [slice(0, 0)]
        if keys.shape[0] == 0
        else [
            slice(start, min(start + batch_rows, keys.shape[0]))
            for start in range(0, keys.shape[0], batch_rows)
        ]
    )
    for row_slice in slices:
        batch = build_row_source_signatures(
            stage=INCREMENTAL_CROP_SIGNATURE_STAGE,
            instance_keys=keys[row_slice],
            content_components={
                "bbox_norm_coords": boxes[row_slice],
                "frame_indices": frame_indices[row_slice],
            },
            compatibility_context=context,
        )
        if expected_spec is None:
            expected_spec = batch.spec
        elif batch.spec.spec_digest != expected_spec.spec_digest:
            raise IncrementalCropError(
                "Crop signature specification changed between batches."
            )
        signatures[row_slice] = batch.signatures
    assert expected_spec is not None
    rowset = build_group_rowset_fingerprint(
        source_group,
        source_rowset_path=source_label,
    )
    if not rowset.is_complete:
        raise IncrementalCropError(
            "Incremental crops require a complete keyed source rowset."
        )
    return CropSourceSnapshot(
        source_path=source_label,
        instance_keys=keys,
        frame_indices=frame_indices,
        bbox_norm_coords=boxes,
        optional_row_arrays=optional_row_arrays,
        signatures=signatures,
        signature_spec=expected_spec,
        rowset_fingerprint=rowset,
    )


def _validate_frame_source(
    frame_source: Any, snapshot: CropSourceSnapshot
) -> tuple[int, int]:
    shape = tuple(int(value) for value in getattr(frame_source, "shape", ()))
    if len(shape) != 3:
        raise IncrementalCropError(
            "Phase-1 incremental crop input must be grayscale uint8 [frame,height,width]."
        )
    if np.dtype(getattr(frame_source, "dtype", None)) != np.dtype(np.uint8):
        raise IncrementalCropError(
            "Phase-1 incremental crop input must use uint8 dtype."
        )
    if snapshot.frame_indices.size and int(snapshot.frame_indices.max()) >= shape[0]:
        raise IncrementalCropError(
            "Source frame_indices exceeds the source-frame array."
        )
    return _normalize_frame_shape((shape[1], shape[2]))


def _validate_base_crop(base_group: Any) -> RowSourceSignatureSpec:
    if base_group.attrs.get(RUN_COMPLETION_STATUS_ATTR) != RUN_STATUS_COMPLETE:
        raise IncrementalCropError("Reuse source crop run is not explicitly complete.")
    missing = [
        name
        for name in ("roi_images", "instance_key", ROW_SOURCE_SIGNATURE_ARRAY)
        if name not in base_group
    ]
    if missing:
        raise IncrementalCropError(
            "Reuse source is not a Phase-1 materialized crop run; missing "
            + ", ".join(missing)
        )
    if str(base_group.attrs.get("crop_storage_mode", "")).strip() != "materialized":
        raise IncrementalCropError("Reuse source crop run is not materialized.")
    roi_shape = tuple(int(value) for value in base_group["roi_images"].shape)
    if len(roi_shape) != 3:
        raise IncrementalCropError("Reuse source ROI payload is not three-dimensional.")
    if np.dtype(base_group["roi_images"].dtype) != np.dtype(np.uint8):
        raise IncrementalCropError("Reuse source ROI payload must use uint8 dtype.")
    base_keys = np.asarray(base_group["instance_key"][:], dtype=np.uint64).reshape(-1)
    if roi_shape[0] != base_keys.shape[0]:
        raise IncrementalCropError("Reuse source identity and ROI row counts differ.")
    if int(np.unique(base_keys).shape[0]) != int(base_keys.shape[0]):
        raise IncrementalCropError(
            "Reuse source contains duplicate instance_key values."
        )
    validate_row_source_signature_array(
        base_group[ROW_SOURCE_SIGNATURE_ARRAY],
        expected_row_count=int(base_keys.shape[0]),
    )
    return load_row_source_signature_spec(base_group.attrs)


def build_incremental_crop_plan(
    snapshot: CropSourceSnapshot,
    *,
    base_group: Any | None,
    roi_size: tuple[int, int],
) -> KeyedDeltaPlan:
    """Build a crop plan, refusing compatibility reuse from legacy bases."""

    if base_group is None:
        return build_keyed_delta_plan(
            target_instance_keys=snapshot.instance_keys,
            target_source_signatures=snapshot.signatures,
            target_signature_spec_digest=snapshot.signature_spec.spec_digest,
        )
    base_spec = _validate_base_crop(base_group)
    plan = build_keyed_delta_plan(
        target_instance_keys=snapshot.instance_keys,
        target_source_signatures=snapshot.signatures,
        target_signature_spec_digest=snapshot.signature_spec.spec_digest,
        source_instance_keys=np.asarray(base_group["instance_key"][:], dtype=np.uint64),
        source_row_signatures=np.asarray(
            base_group[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8
        ),
        source_signature_spec_digest=base_spec.spec_digest,
    )
    if np.any(plan.action_codes != ACTION_CODE_MAP["compute"]):
        base_roi_shape = tuple(
            int(value) for value in base_group["roi_images"].shape[1:]
        )
        if base_roi_shape != _normalize_roi_size(roi_size):
            raise IncrementalCropError(
                "Reuse plan selected crop rows whose base ROI shape differs from the target."
            )
    return plan


def _roi_top_left(
    bbox_norm_coords: np.ndarray,
    *,
    frame_shape: tuple[int, int],
    roi_size: tuple[int, int],
) -> np.ndarray:
    frame_height, frame_width = frame_shape
    roi_height, roi_width = roi_size
    centers = np.round(
        np.asarray(bbox_norm_coords[:, :2], dtype=np.float32)
        * np.asarray([frame_width, frame_height], dtype=np.float32)
    ).astype(np.int32, copy=False)
    coordinates = np.empty((centers.shape[0], 2), dtype=np.int32)
    coordinates[:, 0] = centers[:, 0] - roi_width // 2
    coordinates[:, 1] = centers[:, 1] - roi_height // 2
    return coordinates


def _canonical_crop_coordinate_arrays(
    snapshot: CropSourceSnapshot,
    *,
    frame_shape: tuple[int, int],
    roi_size: tuple[int, int],
    roi_coordinates_full: np.ndarray,
    source_geometry: BoundDetectionObservationGeometry,
) -> dict[str, np.ndarray]:
    """Prepare future-canonical crop geometry without dimensional inference."""

    values = detection_observation_geometry_values(source_geometry)
    source = source_geometry
    if source.row_identity.rowset_path != snapshot.source_path:
        raise IncrementalCropError(
            "Canonical coordinate evidence does not bind the selected source_path."
        )
    if not np.array_equal(values["instance_key"], snapshot.instance_keys):
        raise IncrementalCropError(
            "Canonical source identity differs from the captured crop rowset."
        )
    if not np.array_equal(values["bbox_norm_coords"], snapshot.bbox_norm_coords):
        raise IncrementalCropError(
            "Canonical normalized bbox values differ from the captured crop rowset."
        )
    if not np.array_equal(
        values["source_acquisition_frame_index"],
        snapshot.frame_indices,
    ):
        raise IncrementalCropError(
            "Future-canonical incremental crop v1 requires frame_indices to be "
            "the exact acquisition-frame mapping; local/sparse frame domains need "
            "an explicit frame-selection contract."
        )
    camera = source.frame_evidence.bbox_source_camera_frame
    expected_frame_shape = (int(camera.endpoint.height), int(camera.endpoint.width))
    if frame_shape != expected_frame_shape:
        raise IncrementalCropError(
            "Crop frame source dimensions differ from the exact acquisition-owned "
            f"source-camera frame: {frame_shape} != {expected_frame_shape}."
        )
    roi_height, roi_width = roi_size
    top_left = np.asarray(roi_coordinates_full, dtype=np.int64)
    rounded_centers = np.round(values["centers_img_xy"]).astype(np.int64)
    expected_top_left = np.column_stack(
        (
            rounded_centers[:, 0] - roi_width // 2,
            rounded_centers[:, 1] - roi_height // 2,
        )
    )
    if (
        top_left.shape != (snapshot.row_count, 2)
        or not np.array_equal(top_left, expected_top_left)
        or np.any(top_left[:, 0] < 0)
        or np.any(top_left[:, 1] < 0)
        or np.any(top_left[:, 0] + roi_width > frame_shape[1])
        or np.any(top_left[:, 1] + roi_height > frame_shape[0])
    ):
        raise IncrementalCropError(
            "Future-canonical crop placement v1 requires exact persisted-center "
            "rounding and every ROI to be fully contained in the source camera. "
            "Padded or differently placed crops need an explicit placement lineage "
            "contract and cannot be relabeled."
        )
    bbox_img = values["bbox_img_xyxy"]
    placement_dtype = bbox_img.dtype
    source_crop_xywh = np.column_stack(
        (
            top_left[:, 0],
            top_left[:, 1],
            np.full(snapshot.row_count, roi_width, dtype=np.int64),
            np.full(snapshot.row_count, roi_height, dtype=np.int64),
        )
    ).astype(placement_dtype, copy=False)
    offsets = np.column_stack(
        (
            source_crop_xywh[:, 0],
            source_crop_xywh[:, 1],
            source_crop_xywh[:, 0],
            source_crop_xywh[:, 1],
        )
    )
    bbox_roi = np.asarray(bbox_img - offsets, dtype=bbox_img.dtype)
    return {
        "source_acquisition_frame_index": values["source_acquisition_frame_index"],
        "bbox_img_xyxy": values["bbox_img_xyxy"],
        "centers_img_xy": values["centers_img_xy"],
        "source_crop_xywh": np.array(source_crop_xywh, copy=True, order="C"),
        "bbox_roi_xyxy": np.array(bbox_roi, copy=True, order="C"),
    }


def _canonical_roi_top_left(
    source_geometry: BoundDetectionObservationGeometry,
    *,
    roi_size: tuple[int, int],
) -> np.ndarray:
    """Place crops from the exact persisted source-camera center surface."""

    values = detection_observation_geometry_values(source_geometry)
    centers = values["centers_img_xy"]
    if centers.dtype.kind != "f" or centers.ndim != 2 or centers.shape[1:] != (2,):
        raise IncrementalCropError(
            "Canonical source centers must be one floating source-camera (N,2) array."
        )
    rounded = np.round(centers).astype(np.int64)
    roi_height, roi_width = roi_size
    coordinates = np.empty(rounded.shape, dtype=np.int32)
    coordinates[:, 0] = rounded[:, 0] - roi_width // 2
    coordinates[:, 1] = rounded[:, 1] - roi_height // 2
    return coordinates


def _publish_canonical_crop_coordinate_contract(
    run_group: Any,
    roi_images: Any,
    bbox_edge_frame_node: Any,
    *,
    output_name: str,
    source_geometry: BoundDetectionObservationGeometry,
) -> None:
    """Publish and reload the complete source-camera/ROI crop contract."""

    crop = publish_crop_observation_geometry(
        run_group,
        run_group["instance_key"],
        run_group["detection_indices"],
        run_group["source_acquisition_frame_index"],
        run_group["bbox_norm_coords"],
        run_group["bbox_img_xyxy"],
        run_group["centers_img_xy"],
        source_geometry=source_geometry,
    )
    source = crop.source_geometry
    placement_node = run_group["source_crop_xywh"]
    point_ownership = stamp_crop_placement_ownership(
        placement_node,
        row_identity=crop.row_identity,
        source_camera_frame=source.frame_evidence.source_camera_frame,
    )
    token = hashlib.sha256(output_name.encode("utf-8")).hexdigest()[:16]
    point_roi_frame = stamp_roi_pixel_frame_authority(
        bind_array_reference_extent(roi_images, units="px"),
        frame_id=f"crop_roi_continuous_{token}",
        pixel_convention="continuous",
        crop_placement_ownership=point_ownership,
    )
    point_authority = stamp_crop_placement_transform_authority(
        placement_node,
        authority_id=f"crop_roi_continuous_to_source_camera_{token}",
        source_frame=point_roi_frame,
        target_frame=source.frame_evidence.source_camera_frame,
    )
    stamp_directed_transform_v2(
        placement_node,
        transform_id=f"crop_roi_continuous_to_source_camera_{token}",
        authority=point_authority,
        source_frame=point_roi_frame,
        target_frame=source.frame_evidence.source_camera_frame,
        row_identity=crop.row_identity,
    )

    bbox_ownership = stamp_crop_placement_ownership(
        placement_node,
        row_identity=crop.row_identity,
        source_camera_frame=source.frame_evidence.bbox_source_camera_frame,
        attr_name=CROP_PLACEMENT_PIXEL_EDGE_OWNERSHIP_ATTR,
    )
    bbox_roi_frame = stamp_roi_pixel_frame_authority(
        publish_crop_roi_bbox_edge_reference_extent(
            bbox_edge_frame_node,
            roi_images,
        ),
        frame_id=f"crop_roi_bbox_edge_{token}",
        pixel_convention="pixel_edge_half_open",
        crop_placement_ownership=bbox_ownership,
    )
    bbox_authority = stamp_crop_placement_transform_authority(
        placement_node,
        authority_id=f"crop_roi_bbox_edge_to_source_camera_{token}",
        source_frame=bbox_roi_frame,
        target_frame=source.frame_evidence.bbox_source_camera_frame,
        attr_name=TRANSFORM_AUTHORITY_PIXEL_EDGE_ATTR,
    )
    bbox_transform = stamp_directed_transform_v2(
        placement_node,
        transform_id=f"crop_roi_bbox_edge_to_source_camera_{token}",
        authority=bbox_authority,
        source_frame=bbox_roi_frame,
        target_frame=source.frame_evidence.bbox_source_camera_frame,
        row_identity=crop.row_identity,
        attr_name=DIRECTED_TRANSFORM_V2_PIXEL_EDGE_ATTR,
    )
    publish_crop_roi_geometry(
        placement_node,
        run_group["bbox_roi_xyxy"],
        crop_geometry=crop,
        crop_placement_ownership=bbox_ownership,
        roi_frame=bbox_roi_frame,
        roi_to_source_camera=resolve_bound_directed_transform_chain(
            (bbox_transform,)
        ),
        roi_top_left_node=run_group["roi_coordinates_full"],
        roi_top_left_placement_ownership=point_ownership,
    )
    run_group.attrs["coordinate_contract"] = "canonical_v2"


def _crop_one(
    frame: np.ndarray,
    *,
    top_left: np.ndarray,
    roi_size: tuple[int, int],
) -> np.ndarray:
    roi_height, roi_width = roi_size
    frame_height, frame_width = frame.shape
    x1, y1 = int(top_left[0]), int(top_left[1])
    x2, y2 = x1 + roi_width, y1 + roi_height
    output = np.zeros((roi_height, roi_width), dtype=np.uint8)
    source_x1, source_y1 = max(0, x1), max(0, y1)
    source_x2, source_y2 = min(frame_width, x2), min(frame_height, y2)
    if source_x2 > source_x1 and source_y2 > source_y1:
        output_x1, output_y1 = max(0, -x1), max(0, -y1)
        output_x2 = output_x1 + source_x2 - source_x1
        output_y2 = output_y1 + source_y2 - source_y1
        output[output_y1:output_y2, output_x1:output_x2] = frame[
            source_y1:source_y2, source_x1:source_x2
        ]
    return output


def _copy_base_rows(
    base_array: Any,
    source_rows: np.ndarray,
    output: np.ndarray,
    output_rows: np.ndarray,
) -> int:
    """Copy arbitrary keyed rows using contiguous source reads where possible."""

    if source_rows.size == 0:
        return 0
    order = np.argsort(source_rows, kind="stable")
    sorted_source = source_rows[order]
    sorted_output = output_rows[order]
    start = 0
    bytes_read = 0
    while start < sorted_source.shape[0]:
        stop = start + 1
        while (
            stop < sorted_source.shape[0]
            and sorted_source[stop] == sorted_source[stop - 1] + 1
        ):
            stop += 1
        source_start = int(sorted_source[start])
        source_stop = int(sorted_source[stop - 1]) + 1
        payload = np.asarray(base_array[source_start:source_stop], dtype=np.uint8)
        output[sorted_output[start:stop]] = payload
        bytes_read += int(payload.nbytes)
        start = stop
    return bytes_read


def _write_compact_arrays(
    run_group: Any,
    *,
    snapshot: CropSourceSnapshot,
    roi_coordinates_full: np.ndarray,
    frame_count: int,
    downsampled_frame_shape: tuple[int, int] | None,
    tabular_shard_rows: int,
    canonical_coordinate_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "instance_key": snapshot.instance_keys,
        "frame_indices": snapshot.frame_indices,
        "bbox_norm_coords": snapshot.bbox_norm_coords,
        "detection_indices": np.arange(snapshot.row_count, dtype=np.int64),
        "roi_coordinates_full": roi_coordinates_full,
        ROW_SOURCE_SIGNATURE_ARRAY: snapshot.signatures,
        "frame_counts": np.bincount(
            snapshot.frame_indices, minlength=int(frame_count)
        ).astype(np.int64, copy=False),
    }
    if downsampled_frame_shape is not None:
        ds_height, ds_width = _normalize_frame_shape(downsampled_frame_shape)
        full_height, full_width = snapshot.signature_spec.compatibility_context[
            "source_frame_shape"
        ]
        scale_x = float(ds_width) / float(full_width)
        scale_y = float(ds_height) / float(full_height)
        coords_ds = np.empty_like(roi_coordinates_full)
        coords_ds[:, 0] = (
            roi_coordinates_full[:, 0].astype(np.float32) * scale_x
        ).astype(np.int32)
        coords_ds[:, 1] = (
            roi_coordinates_full[:, 1].astype(np.float32) * scale_y
        ).astype(np.int32)
        arrays["roi_coordinates_ds"] = coords_ds
    for name, values in snapshot.optional_row_arrays.items():
        arrays[name] = values
    if canonical_coordinate_arrays is not None:
        collisions = set(arrays).intersection(canonical_coordinate_arrays)
        if collisions:
            raise IncrementalCropError(
                "Canonical coordinate arrays collide with existing crop arrays: "
                + ", ".join(sorted(collisions))
            )
        arrays.update(canonical_coordinate_arrays)
    for name, values in arrays.items():
        store_array(
            run_group,
            name,
            np.asarray(values),
            shard_rows=int(tabular_shard_rows),
        )


def _validate_output(
    run_group: Any,
    *,
    snapshot: CropSourceSnapshot,
    expected_coordinates: np.ndarray,
    expected_roi_size: tuple[int, int],
) -> None:
    required = (
        "roi_images",
        "instance_key",
        "frame_indices",
        "bbox_norm_coords",
        "roi_coordinates_full",
        ROW_SOURCE_SIGNATURE_ARRAY,
        "materialization_plan",
    )
    missing = [name for name in required if name not in run_group]
    if missing:
        raise IncrementalCropError(
            "Incremental crop output is incomplete: " + ", ".join(missing)
        )
    if tuple(int(value) for value in run_group["roi_images"].shape) != (
        snapshot.row_count,
        *expected_roi_size,
    ):
        raise IncrementalCropError("Incremental crop ROI output shape is incorrect.")
    if not np.array_equal(
        np.asarray(run_group["instance_key"][:], dtype=np.uint64),
        snapshot.instance_keys,
    ):
        raise IncrementalCropError(
            "Incremental crop output key order differs from the target."
        )
    if not np.array_equal(
        np.asarray(run_group[ROW_SOURCE_SIGNATURE_ARRAY][:], dtype=np.uint8),
        snapshot.signatures,
    ):
        raise IncrementalCropError(
            "Incremental crop output signatures differ from the target."
        )
    if not np.array_equal(
        np.asarray(run_group["roi_coordinates_full"][:], dtype=np.int32),
        expected_coordinates,
    ):
        raise IncrementalCropError(
            "Incremental crop output coordinates differ from the target."
        )
    for name in OPTIONAL_SOURCE_ROW_ARRAYS:
        source_has = name in snapshot.optional_row_arrays
        output_has = name in run_group
        if source_has != output_has:
            raise IncrementalCropError(
                f"Incremental crop output lineage presence differs for {name!r}."
            )
        if source_has and not np.array_equal(
            snapshot.optional_row_arrays[name],
            np.asarray(run_group[name][:]),
        ):
            raise IncrementalCropError(
                f"Incremental crop output lineage differs for {name!r}."
            )


def _assert_parent_state_unchanged(parent: Any, expected: Mapping[str, object]) -> None:
    changed = {
        name: (expected.get(name), parent.attrs.get(name))
        for name in expected
        if parent.attrs.get(name) != expected.get(name)
    }
    if changed:
        raise IncrementalCropError(
            "Crop publication state changed while materialization was running: "
            + repr(changed)
        )


def _snapshot_crop_selectors(parent: Any) -> dict[str, tuple[bool, Any]]:
    attrs = parent.attrs
    return {
        name: (name in attrs, copy.deepcopy(attrs.get(name)))
        for name in _CROP_SELECTOR_ATTRS
    }


def _restore_crop_selectors(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
) -> None:
    attrs = parent.attrs
    for name, (present, value) in snapshot.items():
        if present:
            attrs[name] = copy.deepcopy(value)
        elif name in attrs:
            del attrs[name]
    mismatches = {
        name: (present, value, name in attrs, attrs.get(name))
        for name, (present, value) in snapshot.items()
        if (name in attrs) != present or (present and attrs.get(name) != value)
    }
    if mismatches:
        raise IncrementalCropError(
            "Crop selector rollback was incomplete: " + repr(mismatches)
        )


def _assert_crop_selectors_equal(
    parent: Any,
    snapshot: Mapping[str, tuple[bool, Any]],
) -> None:
    attrs = parent.attrs
    changed = {
        name: (present, value, name in attrs, attrs.get(name))
        for name, (present, value) in snapshot.items()
        if (name in attrs) != present or (present and attrs.get(name) != value)
    }
    if changed:
        raise IncrementalCropError(
            "Crop selector state changed during historical materialization: "
            + repr(changed)
        )


def _rollback_failed_crop_publication(
    *,
    run_group: Any,
    parent: Any,
    run_name: str,
    selector_snapshot: Mapping[str, tuple[bool, Any]],
    coordinate_checkpoint: Any | None,
    cause: BaseException,
) -> None:
    failures: list[str] = []
    if coordinate_checkpoint is not None:
        try:
            restore_observation_coordinate_publication_checkpoint(
                coordinate_checkpoint,
                cause=cause,
            )
        except BaseException as exc:
            failures.append(f"coordinate attrs: {exc}")
    try:
        run_group.attrs.update(
            {
                "status": "failed",
                "validation_status": "failed",
                "stage_selector_eligible": False,
            }
        )
        if RUN_COMPLETED_AT_ATTR in run_group.attrs:
            del run_group.attrs[RUN_COMPLETED_AT_ATTR]
        mark_run_failed(
            run_group,
            parent_group=None,
            run_name=run_name,
            error=f"{type(cause).__name__}: {cause}",
        )
        if run_group.attrs.get("stage_selector_eligible") is not False:
            raise IncrementalCropError(
                "Failed crop remained eligible for normal stage selection."
            )
    except BaseException as exc:  # pragma: no cover - hostile persistent mapping
        failures.append(f"failed-state marker: {exc}")
    try:
        _restore_crop_selectors(parent, selector_snapshot)
    except BaseException as exc:  # pragma: no cover - hostile persistent mapping
        failures.append(f"selectors: {exc}")
    if failures:
        raise IncrementalCropError(
            "Crop publication failed and rollback was incomplete: "
            f"{failures!r}."
        ) from cause


def materialize_incremental_crop_run(
    root: Any,
    *,
    source_group: Any,
    source_path: str,
    frame_source: Any,
    source_pixel_fingerprint: str,
    roi_size: tuple[int, int],
    run_name: str,
    run_provenance: Mapping[str, Any],
    base_run_name: str | None = None,
    downsampled_frame_shape: tuple[int, int] | None = None,
    roi_chunk_rows: int = DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    signature_batch_rows: int = DEFAULT_SIGNATURE_BATCH_ROWS,
    tabular_shard_rows: int = DEFAULT_TABULAR_SHARD_ROWS,
    before_publish: Callable[[], None] | None = None,
    source_coordinate_geometry: BoundDetectionObservationGeometry | None = None,
    coordinate_contract_mode: str = "canonical",
) -> IncrementalCropResult:
    """Write, validate, and atomically select one complete crop replacement.

    Publication assumes the repository's serialized-parent-writer contract.
    Under that ownership rule, all selection pointers are changed in one attrs
    update only after the source and expected prior publication state are
    revalidated.  Any exception leaves the prior selection unchanged.
    """

    output_name = str(run_name).strip()
    if not output_name or "/" in output_name:
        raise IncrementalCropError("run_name must be one non-empty group name.")
    if source_coordinate_geometry is not None:
        raise IncrementalCropError(
            "Detached source_coordinate_geometry injection is unsupported. "
            "Canonical crop geometry is always loaded from the exact persisted "
            "selected detect_runs rowset."
        )
    if coordinate_contract_mode != "canonical":
        raise IncrementalCropError(
            "Ordinary materialized crop_runs are canonical-only; legacy or "
            "unbound materialized output is forbidden."
        )
    roi_shape = _normalize_roi_size(roi_size)
    zarr_format = getattr(getattr(root, "metadata", None), "zarr_format", None)
    if zarr_format != 3:
        raise IncrementalCropError("Incremental crop materialization requires Zarr v3.")
    coordinate_context = resolve_incremental_crop_coordinate_context(
        root,
        source_group=source_group,
        source_path=source_path,
        frame_source=frame_source,
        source_pixel_fingerprint=source_pixel_fingerprint,
        coordinate_contract_mode=coordinate_contract_mode,
    )
    source_group = coordinate_context.source_group
    frame_source = coordinate_context.frame_source
    source_pixel_fingerprint = coordinate_context.source_pixel_fingerprint
    raw_frame_shape = tuple(int(value) for value in frame_source.shape[1:])
    snapshot = capture_crop_source_snapshot(
        source_group,
        source_path=source_path,
        source_pixel_fingerprint=source_pixel_fingerprint,
        frame_shape=raw_frame_shape,
        roi_size=roi_shape,
        signature_batch_rows=signature_batch_rows,
    )
    frame_shape = _validate_frame_source(frame_source, snapshot)
    effective_source_coordinate_geometry = coordinate_context.source_geometry
    parent = require_runs_parent(root, "crop_runs")
    if output_name in parent:
        raise IncrementalCropError(f"Crop run {output_name!r} already exists.")
    base_group = None
    if base_run_name is not None:
        base_name = str(base_run_name).strip()
        if not base_name or base_name not in parent:
            raise IncrementalCropError(
                f"Reuse source crop run {base_run_name!r} does not exist."
            )
        base_group = parent[base_name]
    plan = build_incremental_crop_plan(
        snapshot, base_group=base_group, roi_size=roi_shape
    )
    roi_coordinates = _roi_top_left(
        snapshot.bbox_norm_coords,
        frame_shape=frame_shape,
        roi_size=roi_shape,
    )
    canonical_coordinate_arrays = None
    if effective_source_coordinate_geometry is not None:
        roi_coordinates = _canonical_roi_top_left(
            effective_source_coordinate_geometry,
            roi_size=roi_shape,
        )
        canonical_coordinate_arrays = _canonical_crop_coordinate_arrays(
            snapshot,
            frame_shape=frame_shape,
            roi_size=roi_shape,
            roi_coordinates_full=roi_coordinates,
            source_geometry=effective_source_coordinate_geometry,
        )
    provenance_result = validate_run_provenance(run_provenance)
    if not provenance_result.valid:
        raise IncrementalCropError(
            "Invalid run provenance: " + "; ".join(provenance_result.errors)
        )
    normalized_provenance = provenance_result.normalized or dict(run_provenance)
    expected_parent_state = {
        "latest": parent.attrs.get("latest"),
        RUN_LATEST_COMPLETE_ATTR: parent.attrs.get(RUN_LATEST_COMPLETE_ATTR),
        "latest_materialized": parent.attrs.get("latest_materialized"),
        "latest_any": parent.attrs.get("latest_any"),
        "publication_generation": parent.attrs.get("publication_generation"),
    }
    selector_snapshot = _snapshot_crop_selectors(parent)
    run_group = parent.create_group(output_name)
    run_group.attrs["stage_selector_eligible"] = False
    mark_run_started(run_group, run_name=output_name, stage="crop")
    note_pending_latest(parent, output_name)
    coordinate_checkpoint = None
    payload_bytes_written = 0
    base_bytes_read = 0
    frame_bytes_read = 0
    readback_bytes = 0
    source_family = snapshot.source_path.split("/", maxsplit=1)[0]
    detection_source_type = (
        "refined" if source_family.startswith("refined_detect") else "detect"
    )
    try:
        run_group.attrs.update(
            {
                "schema_id": INCREMENTAL_CROP_SCHEMA_ID,
                "schema_version": INCREMENTAL_CROP_SCHEMA_VERSION,
                "stage": "crop",
                "status": "running",
                "crop_storage_mode": "materialized",
                "crop_revision": 0,
                "total_detections": snapshot.row_count,
                "total_frames": int(frame_source.shape[0]),
                "height": frame_shape[0],
                "width": frame_shape[1],
                "roi_size": list(roi_shape),
                "detection_source_path": snapshot.source_path,
                "detection_source_type": detection_source_type,
                "video_source_type": "zarr",
                "source_coords_path": snapshot.source_path,
                "source_pixel_fingerprint": str(source_pixel_fingerprint),
                "coordinate_contract_mode": coordinate_contract_mode,
                "reuse_source_crop_run": base_run_name,
                "writer_ownership": INCREMENTAL_CROP_WRITER_OWNERSHIP,
                "publication_policy": INCREMENTAL_CROP_PUBLICATION_POLICY,
                "tabular_shard_rows_requested": int(tabular_shard_rows),
                "signature_batch_rows": int(signature_batch_rows),
                RUN_PROVENANCE_ATTR: normalized_provenance,
                **snapshot.signature_spec.to_attrs(),
                **snapshot.rowset_fingerprint.to_attrs(),
            }
        )
        if effective_source_coordinate_geometry is None:
            raise IncrementalCropError(
                "Canonical crop source geometry is unavailable."
            )
        pixel_contract = crop_run_pixel_contract(
            crop_storage_mode="materialized",
            video_source_type="zarr",
            acceleration="cpu",
        )
        run_group.attrs.update(
            {
                "roi_image_representation": pixel_contract["image_representation"],
                "roi_pixel_contract": pixel_contract,
                "roi_pixel_contract_name": pixel_contract["name"],
                "center_rounding": CENTER_ROUNDING_NP_ROUND,
            }
        )
        _write_compact_arrays(
            run_group,
            snapshot=snapshot,
            roi_coordinates_full=roi_coordinates,
            frame_count=int(frame_source.shape[0]),
            downsampled_frame_shape=downsampled_frame_shape,
            tabular_shard_rows=int(tabular_shard_rows),
            canonical_coordinate_arrays=canonical_coordinate_arrays,
        )
        plan_group = run_group.create_group("materialization_plan")
        write_keyed_delta_plan(
            plan_group,
            plan,
            shard_rows=int(tabular_shard_rows),
        )
        roi_layout = build_canonical_crop_roi_layout(
            total_rois=snapshot.row_count,
            preferred_chunk_len=int(roi_chunk_rows),
            roi_storage="compressed",
            use_sharding=False,
        )
        run_group.attrs.update(crop_roi_layout_attrs(roi_layout))
        roi_images = run_group.create_array(
            "roi_images",
            **build_crop_roi_create_kwargs(
                total_rois=snapshot.row_count,
                roi_sz=roi_shape,
                layout=roi_layout,
                overwrite=True,
            ),
        )
        output_chunk_rows = int(roi_images.chunks[0])
        for row_start in range(0, snapshot.row_count, output_chunk_rows):
            row_stop = min(row_start + output_chunk_rows, snapshot.row_count)
            row_count = row_stop - row_start
            output = np.empty((row_count, *roi_shape), dtype=np.uint8)
            actions = plan.action_codes[row_start:row_stop]
            copy_local = np.flatnonzero(actions != ACTION_CODE_MAP["compute"])
            if copy_local.size:
                if base_group is None:
                    raise IncrementalCropError(
                        "Copy plan has no reuse source crop run."
                    )
                source_rows = plan.source_row_indices[row_start:row_stop][copy_local]
                base_bytes_read += _copy_base_rows(
                    base_group["roi_images"],
                    source_rows,
                    output,
                    copy_local,
                )
            compute_local = np.flatnonzero(actions == ACTION_CODE_MAP["compute"])
            frame_cache: dict[int, np.ndarray] = {}
            for local_row in compute_local:
                target_row = row_start + int(local_row)
                frame_index = int(snapshot.frame_indices[target_row])
                frame = frame_cache.get(frame_index)
                if frame is None:
                    frame = np.asarray(frame_source[frame_index], dtype=np.uint8)
                    if frame.shape != frame_shape:
                        raise IncrementalCropError(
                            "Source frame shape changed during processing."
                        )
                    frame_cache[frame_index] = frame
                    frame_bytes_read += int(frame.nbytes)
                output[local_row] = _crop_one(
                    frame,
                    top_left=roi_coordinates[target_row],
                    roi_size=roi_shape,
                )
            roi_images[row_start:row_stop] = output
            payload_bytes_written += int(output.nbytes)
            persisted = np.asarray(roi_images[row_start:row_stop], dtype=np.uint8)
            readback_bytes += int(persisted.nbytes)
            if not np.array_equal(persisted, output):
                raise IncrementalCropError(
                    "ROI payload readback differs from the written chunk."
                )

        _validate_output(
            run_group,
            snapshot=snapshot,
            expected_coordinates=roi_coordinates,
            expected_roi_size=roi_shape,
        )
        if before_publish is not None:
            before_publish()
        refreshed_coordinate_context = resolve_incremental_crop_coordinate_context(
            root,
            source_group=root[snapshot.source_path],
            source_path=snapshot.source_path,
            frame_source=frame_source,
            source_pixel_fingerprint=source_pixel_fingerprint,
            coordinate_contract_mode=coordinate_contract_mode,
        )
        if (
            refreshed_coordinate_context.authority_signature
            != coordinate_context.authority_signature
        ):
            raise IncrementalCropError(
                "Acquisition or detection coordinate authority changed during "
                "crop materialization."
            )
        refreshed_source_group = refreshed_coordinate_context.source_group
        refreshed = capture_crop_source_snapshot(
            refreshed_source_group,
            source_path=source_path,
            source_pixel_fingerprint=source_pixel_fingerprint,
            frame_shape=frame_shape,
            roi_size=roi_shape,
            signature_batch_rows=signature_batch_rows,
        )
        assert_rowset_fingerprint_matches(
            snapshot.rowset_fingerprint,
            refreshed.rowset_fingerprint,
            require_complete=True,
        )
        if not np.array_equal(
            snapshot.instance_keys, refreshed.instance_keys
        ) or not np.array_equal(snapshot.signatures, refreshed.signatures):
            raise IncrementalCropError(
                "Source row identity or content changed during processing."
            )
        if (
            snapshot.optional_row_arrays.keys() != refreshed.optional_row_arrays.keys()
            or any(
                not np.array_equal(values, refreshed.optional_row_arrays[name])
                for name, values in snapshot.optional_row_arrays.items()
            )
        ):
            raise IncrementalCropError("Source row lineage changed during processing.")
        publication_parent = root["crop_runs"]
        _assert_parent_state_unchanged(publication_parent, expected_parent_state)
        effective_source_coordinate_geometry = (
            refreshed_coordinate_context.source_geometry
        )
        if effective_source_coordinate_geometry is None:
            raise IncrementalCropError(
                "Canonical crop geometry disappeared before publication."
            )
        summary = {
            **plan.summary(),
            "roi_payload_bytes_written": int(payload_bytes_written),
            "roi_payload_bytes_read_from_base": int(base_bytes_read),
            "source_frame_bytes_read": int(frame_bytes_read),
            "validation_readback_bytes": int(readback_bytes),
        }
        frames_with_crops = int(np.unique(snapshot.frame_indices).shape[0])
        percent_frames = (
            (frames_with_crops / int(frame_source.shape[0])) * 100.0
            if int(frame_source.shape[0]) > 0
            else 0.0
        )
        detection_source = snapshot.optional_row_arrays.get("detection_source")
        interpolated_rows = (
            int(np.count_nonzero(detection_source == 1))
            if detection_source is not None
            else 0
        )
        run_group.attrs.update(
            {
                "materialization_summary": summary,
                "summary_statistics": {
                    "total_frames": int(frame_source.shape[0]),
                    "frames_with_crops": frames_with_crops,
                    "total_rois_cropped": snapshot.row_count,
                    "percent_frames_with_crops": round(percent_frames, 2),
                    "roi_size": list(roi_shape),
                    "roi_pixels_materialized": True,
                },
                "includes_interpolated": interpolated_rows > 0,
                "n_real_detections": snapshot.row_count - interpolated_rows,
                "n_interpolated_detections": interpolated_rows,
                "status": "completed",
                "validation_status": "passed",
            }
        )
        bbox_edge_frame_node = run_group.require_group(
            "coordinate_frames"
        ).require_group("roi_bbox_edge")
        coordinate_checkpoint = (
            capture_observation_coordinate_publication_checkpoint(
                run_group,
                run_group["instance_key"],
                run_group["detection_indices"],
                run_group["source_acquisition_frame_index"],
                run_group["bbox_norm_coords"],
                run_group["bbox_img_xyxy"],
                run_group["centers_img_xy"],
                run_group["source_crop_xywh"],
                run_group["bbox_roi_xyxy"],
                roi_images,
                bbox_edge_frame_node,
            )
        )
        _publish_canonical_crop_coordinate_contract(
            run_group,
            roi_images,
            bbox_edge_frame_node,
            output_name=output_name,
            source_geometry=effective_source_coordinate_geometry,
        )
        mark_run_complete(
            run_group,
            run_name=output_name,
            run_provenance=normalized_provenance,
        )
        _load_persisted_ordinary_crop_observation_geometry(
            root,
            f"crop_runs/{output_name}",
            require_selector_eligible=False,
        )
        previous_generation = expected_parent_state["publication_generation"]
        generation = 1 if previous_generation is None else int(previous_generation) + 1
        result = IncrementalCropResult(
            run_name=output_name,
            plan=plan,
            copied_rows=int(
                np.count_nonzero(plan.action_codes == ACTION_CODE_MAP["copy"])
            ),
            computed_rows=int(
                np.count_nonzero(plan.action_codes == ACTION_CODE_MAP["compute"])
            ),
            omitted_rows=int(plan.omitted_instance_keys.shape[0]),
            roi_payload_bytes_written=int(payload_bytes_written),
            roi_payload_bytes_read_from_base=int(base_bytes_read),
            source_frame_bytes_read=int(frame_bytes_read),
            validation_readback_bytes=int(readback_bytes),
        )
        publication_parent.attrs.update(
            {
                "latest": output_name,
                RUN_LATEST_COMPLETE_ATTR: output_name,
                "latest_materialized": output_name,
                "latest_any": output_name,
                "publication_generation": generation,
                "publication_policy": INCREMENTAL_CROP_PUBLICATION_POLICY,
            }
        )
        if publication_parent.attrs.get(RUN_LATEST_PENDING_ATTR) == output_name:
            del publication_parent.attrs[RUN_LATEST_PENDING_ATTR]
        run_group.attrs["stage_selector_eligible"] = True
        return result
    except BaseException as exc:
        _rollback_failed_crop_publication(
            run_group=run_group,
            parent=root["crop_runs"],
            run_name=output_name,
            selector_snapshot=selector_snapshot,
            coordinate_checkpoint=coordinate_checkpoint,
            cause=exc,
        )
        raise


def materialize_composite_incremental_crop_run(
    root: Any,
    *,
    source_group: Any,
    source_path: str,
    frame_source: Any,
    source_pixel_fingerprint: str,
    roi_size: tuple[int, int],
    run_name: str,
    run_provenance: Mapping[str, Any],
    base_run_name: str,
    downsampled_frame_shape: tuple[int, int] | None = None,
    roi_chunk_rows: int = DEFAULT_CANONICAL_CROP_ROI_CHUNK_LEN,
    signature_batch_rows: int = DEFAULT_SIGNATURE_BATCH_ROWS,
    tabular_shard_rows: int = DEFAULT_TABULAR_SHARD_ROWS,
    promote: bool = False,
    before_publish: Callable[[], None] | None = None,
    coordinate_contract_mode: str = HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE,
) -> IncrementalCropResult:
    """Write a depth-one crop delta that resolves against one immutable base.

    The run contains a complete compact target rowset, but only newly computed
    ROI pixels. It is a historical migration surface, remains explicitly
    ineligible for stage selection, and never changes a normal crop pointer.
    """

    output_name = str(run_name).strip()
    base_name = str(base_run_name).strip()
    if not output_name or "/" in output_name:
        raise IncrementalCropError("run_name must be one non-empty group name.")
    if not base_name or "/" in base_name:
        raise IncrementalCropError("base_run_name must be one non-empty group name.")
    if coordinate_contract_mode != HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE:
        raise IncrementalCropError(
            "Composite crop publication is historical migration-only and requires "
            f"coordinate_contract_mode={HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE!r}."
        )
    if promote:
        raise IncrementalCropError(
            "Historical composite crops are never selector-eligible; promotion "
            "is forbidden."
        )
    roi_shape = _normalize_roi_size(roi_size)
    zarr_format = getattr(getattr(root, "metadata", None), "zarr_format", None)
    if zarr_format != 3:
        raise IncrementalCropError("Composite crop materialization requires Zarr v3.")

    coordinate_context = resolve_historical_composite_coordinate_context(
        root,
        source_group=source_group,
        source_path=source_path,
        frame_source=frame_source,
        source_pixel_fingerprint=source_pixel_fingerprint,
        coordinate_contract_mode=coordinate_contract_mode,
    )
    source_group = coordinate_context.source_group
    frame_source = coordinate_context.frame_source
    source_pixel_fingerprint = coordinate_context.source_pixel_fingerprint
    raw_frame_shape = tuple(int(value) for value in frame_source.shape[1:])
    snapshot = capture_crop_source_snapshot(
        source_group,
        source_path=source_path,
        source_pixel_fingerprint=source_pixel_fingerprint,
        frame_shape=raw_frame_shape,
        roi_size=roi_shape,
        signature_batch_rows=signature_batch_rows,
    )
    frame_shape = _validate_frame_source(frame_source, snapshot)
    parent = require_runs_parent(root, "crop_runs")
    if output_name in parent:
        raise IncrementalCropError(f"Crop run {output_name!r} already exists.")
    if base_name not in parent:
        raise IncrementalCropError(
            f"Reuse source crop run {base_name!r} does not exist."
        )
    base_group = parent[base_name]
    plan = build_incremental_crop_plan(
        snapshot,
        base_group=base_group,
        roi_size=roi_shape,
    )
    provenance_result = validate_run_provenance(run_provenance)
    if not provenance_result.valid:
        raise IncrementalCropError(
            "Invalid run provenance: " + "; ".join(provenance_result.errors)
        )
    normalized_provenance = provenance_result.normalized or dict(run_provenance)
    selector_snapshot = _snapshot_crop_selectors(parent)
    run_group = parent.create_group(output_name)
    run_group.attrs["stage_selector_eligible"] = False
    mark_run_started(
        run_group,
        run_name=output_name,
        stage="crop_historical_composite",
    )

    roi_coordinates = _roi_top_left(
        snapshot.bbox_norm_coords,
        frame_shape=frame_shape,
        roi_size=roi_shape,
    )
    reusable = plan.action_codes != ACTION_CODE_MAP["compute"]
    compute_target_rows = np.flatnonzero(~reusable).astype(np.int64, copy=False)
    source_codes = np.full(
        snapshot.row_count,
        COMPOSITE_CROP_SOURCE_DELTA,
        dtype=np.uint8,
    )
    source_codes[reusable] = COMPOSITE_CROP_SOURCE_BASE
    source_row_indices = np.empty(snapshot.row_count, dtype=np.int64)
    source_row_indices[reusable] = plan.source_row_indices[reusable]
    source_row_indices[~reusable] = np.arange(
        compute_target_rows.shape[0], dtype=np.int64
    )

    payload_bytes_written = 0
    frame_bytes_read = 0
    readback_bytes = 0
    source_family = snapshot.source_path.split("/", maxsplit=1)[0]
    detection_source_type = (
        "refined" if source_family.startswith("refined_detect") else "detect"
    )
    try:
        run_group.attrs.update(
            {
                "schema_id": INCREMENTAL_CROP_SCHEMA_ID,
                "schema_version": INCREMENTAL_CROP_SCHEMA_VERSION,
                "stage": "crop",
                "status": "running",
                "crop_storage_mode": COMPOSITE_CROP_STORAGE_MODE,
                "composite_crop_schema_id": COMPOSITE_CROP_SCHEMA_ID,
                "composite_crop_schema_version": COMPOSITE_CROP_SCHEMA_VERSION,
                "composite_reference_depth": 1,
                "composite_base_crop_run": base_name,
                "crop_revision": 0,
                "total_detections": snapshot.row_count,
                "total_frames": int(frame_source.shape[0]),
                "height": frame_shape[0],
                "width": frame_shape[1],
                "roi_size": list(roi_shape),
                "detection_source_path": snapshot.source_path,
                "detection_source_type": detection_source_type,
                "video_source_type": "zarr",
                "source_coords_path": snapshot.source_path,
                "source_pixel_fingerprint": str(source_pixel_fingerprint),
                "coordinate_contract_mode": coordinate_contract_mode,
                "coordinate_contract": HISTORICAL_COMPOSITE_COORDINATE_CONTRACT,
                "reuse_source_crop_run": base_name,
                "writer_ownership": INCREMENTAL_CROP_WRITER_OWNERSHIP,
                "publication_policy": INCREMENTAL_CROP_PUBLICATION_POLICY,
                "selection_policy": "historical_complete_unselected",
                "tabular_shard_rows_requested": int(tabular_shard_rows),
                "signature_batch_rows": int(signature_batch_rows),
                RUN_PROVENANCE_ATTR: normalized_provenance,
                **snapshot.signature_spec.to_attrs(),
                **snapshot.rowset_fingerprint.to_attrs(),
            }
        )
        pixel_contract = crop_run_pixel_contract(
            crop_storage_mode=COMPOSITE_CROP_STORAGE_MODE,
            video_source_type="zarr",
            acceleration="cpu",
        )
        run_group.attrs.update(
            {
                "roi_image_representation": pixel_contract["image_representation"],
                "roi_pixel_contract": pixel_contract,
                "roi_pixel_contract_name": pixel_contract["name"],
                "center_rounding": CENTER_ROUNDING_NP_ROUND,
            }
        )
        _write_compact_arrays(
            run_group,
            snapshot=snapshot,
            roi_coordinates_full=roi_coordinates,
            frame_count=int(frame_source.shape[0]),
            downsampled_frame_shape=downsampled_frame_shape,
            tabular_shard_rows=int(tabular_shard_rows),
        )
        plan_group = run_group.create_group("materialization_plan")
        write_keyed_delta_plan(
            plan_group,
            plan,
            shard_rows=int(tabular_shard_rows),
        )
        payload = run_group.create_group(COMPOSITE_CROP_PAYLOAD_GROUP)
        store_array(
            payload,
            "source_codes",
            source_codes,
            shard_rows=int(tabular_shard_rows),
        )
        store_array(
            payload,
            "source_row_indices",
            source_row_indices,
            shard_rows=int(tabular_shard_rows),
        )
        store_array(
            payload,
            "delta_target_row_indices",
            compute_target_rows,
            shard_rows=int(tabular_shard_rows),
        )
        store_array(
            payload,
            "delta_instance_key",
            snapshot.instance_keys[compute_target_rows],
            shard_rows=int(tabular_shard_rows),
        )
        roi_layout = build_canonical_crop_roi_layout(
            total_rois=int(compute_target_rows.shape[0]),
            preferred_chunk_len=int(roi_chunk_rows),
            roi_storage="compressed",
            use_sharding=False,
        )
        run_group.attrs.update(
            {
                **crop_roi_layout_attrs(roi_layout),
                "roi_layout_scope": "composite_delta_only",
            }
        )
        delta_images = payload.create_array(
            "roi_images_delta",
            **build_crop_roi_create_kwargs(
                total_rois=int(compute_target_rows.shape[0]),
                roi_sz=roi_shape,
                layout=roi_layout,
                overwrite=True,
            ),
        )
        output_chunk_rows = int(delta_images.chunks[0])
        for delta_start in range(0, compute_target_rows.shape[0], output_chunk_rows):
            delta_stop = min(
                delta_start + output_chunk_rows,
                compute_target_rows.shape[0],
            )
            target_rows = compute_target_rows[delta_start:delta_stop]
            output = np.empty((target_rows.shape[0], *roi_shape), dtype=np.uint8)
            frame_cache: dict[int, np.ndarray] = {}
            for local_row, target_row_value in enumerate(target_rows):
                target_row = int(target_row_value)
                frame_index = int(snapshot.frame_indices[target_row])
                frame = frame_cache.get(frame_index)
                if frame is None:
                    frame = np.asarray(frame_source[frame_index], dtype=np.uint8)
                    if frame.shape != frame_shape:
                        raise IncrementalCropError(
                            "Source frame shape changed during processing."
                        )
                    frame_cache[frame_index] = frame
                    frame_bytes_read += int(frame.nbytes)
                output[local_row] = _crop_one(
                    frame,
                    top_left=roi_coordinates[target_row],
                    roi_size=roi_shape,
                )
            delta_images[delta_start:delta_stop] = output
            payload_bytes_written += int(output.nbytes)
            persisted = np.asarray(delta_images[delta_start:delta_stop], dtype=np.uint8)
            readback_bytes += int(persisted.nbytes)
            if not np.array_equal(persisted, output):
                raise IncrementalCropError(
                    "Composite delta ROI readback differs from the written chunk."
                )

        validate_composite_crop_run(
            parent,
            run_group,
            run_name=output_name,
            require_complete=False,
            verify_identity=True,
        )
        if before_publish is not None:
            before_publish()
        refreshed_coordinate_context = resolve_historical_composite_coordinate_context(
            root,
            source_group=root[snapshot.source_path],
            source_path=snapshot.source_path,
            frame_source=frame_source,
            source_pixel_fingerprint=source_pixel_fingerprint,
            coordinate_contract_mode=coordinate_contract_mode,
        )
        refreshed_source_group = refreshed_coordinate_context.source_group
        refreshed = capture_crop_source_snapshot(
            refreshed_source_group,
            source_path=source_path,
            source_pixel_fingerprint=source_pixel_fingerprint,
            frame_shape=frame_shape,
            roi_size=roi_shape,
            signature_batch_rows=signature_batch_rows,
        )
        assert_rowset_fingerprint_matches(
            snapshot.rowset_fingerprint,
            refreshed.rowset_fingerprint,
            require_complete=True,
        )
        if not np.array_equal(
            snapshot.instance_keys, refreshed.instance_keys
        ) or not np.array_equal(snapshot.signatures, refreshed.signatures):
            raise IncrementalCropError(
                "Source row identity or content changed during processing."
            )
        if (
            snapshot.optional_row_arrays.keys() != refreshed.optional_row_arrays.keys()
            or any(
                not np.array_equal(values, refreshed.optional_row_arrays[name])
                for name, values in snapshot.optional_row_arrays.items()
            )
        ):
            raise IncrementalCropError("Source row lineage changed during processing.")

        publication_parent = root["crop_runs"]
        _assert_crop_selectors_equal(publication_parent, selector_snapshot)
        summary = {
            **plan.summary(),
            "storage_strategy": "immutable_depth_one_base_plus_delta",
            "base_crop_run": base_name,
            "roi_payload_bytes_written": int(payload_bytes_written),
            "roi_payload_bytes_read_from_base": 0,
            "source_frame_bytes_read": int(frame_bytes_read),
            "validation_readback_bytes": int(readback_bytes),
        }
        frames_with_crops = int(np.unique(snapshot.frame_indices).shape[0])
        percent_frames = (
            (frames_with_crops / int(frame_source.shape[0])) * 100.0
            if int(frame_source.shape[0]) > 0
            else 0.0
        )
        detection_source = snapshot.optional_row_arrays.get("detection_source")
        interpolated_rows = (
            int(np.count_nonzero(detection_source == 1))
            if detection_source is not None
            else 0
        )
        run_group.attrs.update(
            {
                "materialization_summary": summary,
                "summary_statistics": {
                    "total_frames": int(frame_source.shape[0]),
                    "frames_with_crops": frames_with_crops,
                    "total_rois_cropped": snapshot.row_count,
                    "percent_frames_with_crops": round(percent_frames, 2),
                    "roi_size": list(roi_shape),
                    "roi_pixels_logically_complete": True,
                    "roi_pixels_materialized": False,
                    "roi_delta_rows_materialized": int(compute_target_rows.shape[0]),
                },
                "includes_interpolated": interpolated_rows > 0,
                "n_real_detections": snapshot.row_count - interpolated_rows,
                "n_interpolated_detections": interpolated_rows,
                "status": "completed",
                "validation_status": "passed",
            }
        )
        mark_run_complete(
            run_group,
            run_name=output_name,
            run_provenance=normalized_provenance,
        )
        validate_composite_crop_run(
            publication_parent,
            run_group,
            run_name=output_name,
            require_complete=True,
            verify_identity=True,
        )
        _assert_crop_selectors_equal(publication_parent, selector_snapshot)
        return IncrementalCropResult(
            run_name=output_name,
            plan=plan,
            copied_rows=int(np.count_nonzero(reusable)),
            computed_rows=int(compute_target_rows.shape[0]),
            omitted_rows=int(plan.omitted_instance_keys.shape[0]),
            roi_payload_bytes_written=int(payload_bytes_written),
            roi_payload_bytes_read_from_base=0,
            source_frame_bytes_read=int(frame_bytes_read),
            validation_readback_bytes=int(readback_bytes),
        )
    except BaseException as exc:
        _rollback_failed_crop_publication(
            run_group=run_group,
            parent=root["crop_runs"],
            run_name=output_name,
            selector_snapshot=selector_snapshot,
            coordinate_checkpoint=None,
            cause=exc,
        )
        raise


__all__ = [
    "INCREMENTAL_CROP_SCHEMA_ID",
    "INCREMENTAL_CROP_SCHEMA_VERSION",
    "INCREMENTAL_CROP_WRITER_OWNERSHIP",
    "INCREMENTAL_CROP_PUBLICATION_POLICY",
    "DEFAULT_SIGNATURE_BATCH_ROWS",
    "DEFAULT_TABULAR_SHARD_ROWS",
    "CROP_COORDINATE_CONTRACT_MODES",
    "HISTORICAL_COMPOSITE_COORDINATE_CONTRACT_MODE",
    "HISTORICAL_COMPOSITE_COORDINATE_CONTRACT",
    "OPTIONAL_SOURCE_ROW_ARRAYS",
    "IncrementalCropError",
    "CropSourceSnapshot",
    "IncrementalCropResult",
    "ResolvedIncrementalCropCoordinateContext",
    "crop_signature_compatibility_context",
    "capture_crop_source_snapshot",
    "build_incremental_crop_plan",
    "resolve_incremental_crop_coordinate_context",
    "resolve_historical_composite_coordinate_context",
    "materialize_incremental_crop_run",
    "materialize_composite_incremental_crop_run",
]
