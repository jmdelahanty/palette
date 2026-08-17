"""Successor-only adapter for the sealed historical geometry-only crop v2.

The ordinary keypoint and subject-mask coordinate loaders intentionally require
an eligible, materialized canonical crop.  A small set of immutable historical
inference runs was produced from the crop-geometry publication instead: its
manifest and row arrays are complete, but its pixels lived in a disposable flat
cache.  This module proves that exact topology for coordinate-only successors.

It never promotes or edits the crop run and never treats the disposable pixel
cache as an input.  The adapter is installed only around successor preparation
and publication; ordinary readers continue to resolve through their existing
strict loaders.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import numpy as np

from fisheye.shared.coordinate_surface_contract import (
    SOURCE_CAMERA_BBOX_PIXEL_CONVENTION,
    SOURCE_CAMERA_POINT_PIXEL_CONVENTION,
)
from fisheye.shared.model_input_transform import ModelInputTransform
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
    load_source_camera_pixel_frame_authority,
)
from fisheye.shared.zarr.crop_manifest import (
    CROP_GEOMETRY_SCHEMA_V1,
    CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION,
    validate_crop_run_manifest,
)
from fisheye.shared.zarr.crop_shadow import (
    open_persisted_crop_geometry_publication,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_SCHEMA_ID = (
    "palette.coordinate_successor.historical_geometry_only_crop_adapter"
)
HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_SCHEMA_VERSION = 1
HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_KIND = (
    "sealed_geometry_only_crop_manifest_v2_no_pixel_source"
)
_LOADER_OVERRIDE_LOCK = RLock()


class HistoricalGeometryOnlyCropAdapterError(ValueError):
    """Raised when a historical geometry-only crop cannot be proven exactly."""


@dataclass(frozen=True)
class HistoricalGeometryOnlyCropBinding:
    """The exact crop evidence and loader-bound compatibility adapter."""

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
    adapter_record: Mapping[str, Any]

    def as_record(self) -> dict[str, Any]:
        return dict(self.adapter_record)


def _array(node: Any, *, label: str) -> np.ndarray:
    try:
        value = np.asarray(node[...])
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise HistoricalGeometryOnlyCropAdapterError(
            f"Historical crop array {label!r} cannot be decoded: {exc}"
        ) from exc
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoricalGeometryOnlyCropAdapterError(
            f"Historical crop {label} must be an object."
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
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop reference run_id does not match the exact crop manifest."
        )
    for name, actual in expected.items():
        if name in value and value.get(name) != actual:
            raise HistoricalGeometryOnlyCropAdapterError(
                f"Historical crop reference {name!r} does not match the exact crop manifest."
            )
    if "manifest_digest" in value and value.get("manifest_digest") != canonical_json_sha256(manifest):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop reference manifest_digest does not match the exact crop manifest."
        )
    if "row_signatures_digest" in value:
        declared = logical.get("document", {}).get("arrays", {}).get(
            "source_row_signature", {}
        )
        if value.get("row_signatures_digest") != declared.get("sha256"):
            raise HistoricalGeometryOnlyCropAdapterError(
                "Historical crop reference row-signatures digest differs from the exact crop manifest."
            )
    declared_dimensions = payload.get("logical_schema", {}).get("dimensions", {})
    reference_dimensions = value.get("dimensions")
    if reference_dimensions is not None:
        if not isinstance(reference_dimensions, Mapping) or any(
            reference_dimensions.get(name) != declared_dimensions.get(name)
            for name in ("n_frames", "n_instances", "source_width", "source_height")
        ):
            raise HistoricalGeometryOnlyCropAdapterError(
                "Historical crop reference dimensions differ from the exact crop manifest."
            )


def _source_dimensions(source_manifest: Mapping[str, Any]) -> tuple[int, int, int]:
    payload = _require_mapping(source_manifest.get("payload"), label="source manifest payload")
    logical = _require_mapping(payload.get("logical_schema"), label="source logical schema")
    dimensions = _require_mapping(logical.get("dimensions"), label="source dimensions")
    values: list[int] = []
    for name in ("n_frames", "n_instances"):
        value = dimensions.get(name)
        if type(value) is not int or value < 0:
            raise HistoricalGeometryOnlyCropAdapterError(
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
        raise HistoricalGeometryOnlyCropAdapterError(
            f"{source_run_path} lacks required crop-selection arrays: {missing!r}."
        )
    rows = _array(source_arrays["source_crop_row_ids"], label="source_crop_row_ids")
    keys = _array(source_arrays["instance_key"], label="source instance_key")
    frames = _array(
        source_arrays["source_acquisition_frame_index"],
        label="source_acquisition_frame_index",
    )
    if rows.dtype != np.dtype("<i8") or rows.shape != (n_instances,):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Source crop row selection has the wrong dtype, shape, or cardinality."
        )
    if keys.dtype != np.dtype("<u8") or keys.shape != (n_instances,):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Source instance_key has the wrong dtype, shape, or cardinality."
        )
    if frames.dtype != np.dtype("<i8") or frames.shape != (n_instances,):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Source acquisition-frame array has the wrong dtype, shape, or cardinality."
        )
    expected_rows = np.arange(n_instances, dtype=np.int64)
    if not np.array_equal(np.sort(rows), expected_rows):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Source crop row selection is not an exact complete permutation of the crop rowset."
        )
    crop_keys = _array(crop_arrays["instance_key"], label="crop instance_key")
    crop_frames = _array(
        crop_arrays["source_acquisition_frame_index"],
        label="crop source_acquisition_frame_index",
    )
    if not np.array_equal(keys, crop_keys[rows]):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Source instance_key values do not match the exact selected crop rows."
        )
    if not np.array_equal(frames, crop_frames[rows]):
        raise HistoricalGeometryOnlyCropAdapterError(
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
            raise HistoricalGeometryOnlyCropAdapterError(
                "Source crop row signatures have the wrong dtype or shape."
            )
        if not np.array_equal(source_signatures, crop_signatures[rows]):
            raise HistoricalGeometryOnlyCropAdapterError(
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
            raise HistoricalGeometryOnlyCropAdapterError(
                "Source crop placement has the wrong dtype or shape."
            )
        if not np.array_equal(source_placement, crop_placement[rows]):
            raise HistoricalGeometryOnlyCropAdapterError(
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
) -> Any:
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
    return SimpleNamespace(
        crop_geometry=crop_geometry,
        roi_geometry=SimpleNamespace(derivation=roi_derivation),
        roi_frame=roi_frame,
        bbox_roi_frame=bbox_roi_frame,
        crop_path=crop_path,
        _root=root,
        _rowset_node=crop,
        _placement_node=crop["source_crop_xywh"],
        _roi_images_node=SimpleNamespace(
            attrs={},
            path=f"{crop_path}/__historical_geometry_only_no_pixel_source__",
        ),
        _seal=None,
    )


def bind_historical_geometry_only_crop_source(
    *,
    analysis_zarr: Path,
    root: Any,
    crop_reference: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_arrays: Mapping[str, Any],
    source_run_path: str,
    model_input_transform: ModelInputTransform,
) -> HistoricalGeometryOnlyCropBinding:
    """Prove and bind the one supported historical geometry-only crop topology."""

    archive = analysis_zarr.expanduser().resolve()
    crop_path = str(crop_reference.get("run_path") or "").strip("/")
    if not crop_path.startswith("crop_runs/") or len(crop_path.split("/")) != 2:
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop adapter requires one exact crop_runs/<run> path."
        )
    run_id = crop_path.split("/", 1)[1]
    publication = open_persisted_crop_geometry_publication(archive, run_id=run_id)
    manifest = dict(publication.manifest)
    errors = validate_crop_run_manifest(manifest)
    if errors:
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop manifest is invalid: " + "; ".join(errors)
        )
    if manifest.get("schema_version") != CROP_COORDINATE_RUN_MANIFEST_SCHEMA_VERSION:
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop adapter requires crop manifest v2 with its coordinate catalog."
        )
    payload = _require_mapping(manifest["payload"], label="manifest payload")
    logical = _require_mapping(payload["logical_schema"], label="logical schema")
    dimensions = _require_mapping(logical["dimensions"], label="dimensions")
    if payload.get("coordinate_contract") is None:
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical geometry-only crop lacks its coordinate catalog."
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
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop does not have the exact complete geometry-only lifecycle."
        )
    child_names = set(crop.keys())
    expected_names = set(CROP_GEOMETRY_SCHEMA_V1.binding_paths)
    if child_names != expected_names or "roi_images" in child_names:
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop arrays do not match the complete geometry-only topology."
        )
    _require_crop_reference(crop_reference, manifest=manifest, crop_path=crop_path)
    n_frames = int(dimensions["n_frames"])
    n_instances = int(dimensions["n_instances"])
    source_width = int(dimensions["source_width"])
    source_height = int(dimensions["source_height"])
    source_n_frames, source_n_instances, _ = _source_dimensions(source_manifest)
    if (source_n_frames, source_n_instances) != (n_frames, n_instances):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Source output dimensions do not match complete crop row coverage."
        )
    crop_arrays = publication.arrays
    expected_array_names = set(CROP_GEOMETRY_SCHEMA_V1.binding_paths)
    if set(crop_arrays) != expected_array_names:
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop publication does not expose the complete exact array set."
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
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop pixel authority dimensions differ from its manifest."
        )
    _, acquisition = load_persisted_acquisition_camera_authority(root)
    camera_id = str(pixel["camera_identity"])
    acquisition_recording_id = getattr(acquisition.record, "recording_id", None)
    if (
        pixel.get("recording_identity") is not None
        and acquisition_recording_id is not None
        and pixel.get("recording_identity") != acquisition_recording_id
    ):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop pixel authority recording differs from acquisition authority."
        )
    if acquisition.record.camera_id != camera_id:
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop pixel authority camera differs from acquisition authority."
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
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop source-camera extent or point/bbox convention is incompatible."
        )
    roi_sizes = _array(crop_arrays["roi_sizes_full"], label="roi_sizes_full")
    if roi_sizes.dtype != np.dtype("<i4") or roi_sizes.shape != (n_instances, 2):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop ROI sizes are not the exact int32 row-aligned array."
        )
    if np.any(roi_sizes <= 0) or not np.all(roi_sizes == np.asarray([roi_sizes[0]])):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop ROI sizes are not one exact constant positive extent."
        )
    roi_width, roi_height = (int(roi_sizes[0, 0]), int(roi_sizes[0, 1]))
    if type(model_input_transform) is not ModelInputTransform:
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop adapter requires one exact model input transform."
        )
    if model_input_transform.native_shape != (roi_height, roi_width):
        raise HistoricalGeometryOnlyCropAdapterError(
            "Historical crop ROI extent differs from the exact model input transform."
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
    declarations = _require_mapping(logical_document["document"], label="logical content document")["arrays"]
    row_digest = _require_mapping(declarations, label="logical content arrays")["source_row_signature"]["sha256"]
    coordinate_catalog_digest = _require_mapping(payload["coordinate_contract"], label="coordinate contract")["digest"]
    return HistoricalGeometryOnlyCropBinding(
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
        adapter_record={
            "schema_id": HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_SCHEMA_ID,
            "schema_version": HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_SCHEMA_VERSION,
            "adapter_kind": HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_KIND,
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
        },
    )


@contextmanager
def historical_geometry_only_crop_loader(
    binding: HistoricalGeometryOnlyCropBinding,
) -> Iterator[None]:
    """Install the adapter for one narrow successor operation only.

    The existing production loaders are ordinary module globals, so this
    compatibility bridge must temporarily replace those two symbols.  The
    replacement is held under one process-local lock, validates the exact root
    and crop path on every call, and is restored by the context manager even
    when the successor operation raises.  No ordinary loader is changed outside
    the ``with`` body.
    """

    from fisheye.shared import keypoint_coordinate_publication as keypoint_publication
    from fisheye.shared import subject_mask_coordinate_publication as mask_publication

    def load(root: Any, crop_path: str) -> Any:
        if root is not binding.source._root or str(crop_path).strip("/") != binding.crop_path:
            raise HistoricalGeometryOnlyCropAdapterError(
                "Historical crop adapter was asked for an unbound root or crop path."
            )
        return binding.source

    with _LOADER_OVERRIDE_LOCK:
        old_keypoint = keypoint_publication.load_persisted_keypoint_crop_source
        old_mask = mask_publication.load_persisted_subject_mask_crop_source
        keypoint_publication.load_persisted_keypoint_crop_source = load
        mask_publication.load_persisted_subject_mask_crop_source = load
        try:
            yield
        finally:
            keypoint_publication.load_persisted_keypoint_crop_source = old_keypoint
            mask_publication.load_persisted_subject_mask_crop_source = old_mask


__all__ = [
    "HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_KIND",
    "HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_SCHEMA_ID",
    "HISTORICAL_GEOMETRY_ONLY_CROP_ADAPTER_SCHEMA_VERSION",
    "HistoricalGeometryOnlyCropAdapterError",
    "HistoricalGeometryOnlyCropBinding",
    "bind_historical_geometry_only_crop_source",
    "historical_geometry_only_crop_loader",
]
